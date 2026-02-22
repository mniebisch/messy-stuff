from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from fmp.datasets.sphere import sampling as sphere_sampling

__all__ = [
    "OnTheFlySphereSliceDataset",
    "FixedSphereSliceDataset",
    "SphereSliceConfig",
    "SphereSliceRanges",
    "make_slice_and_unet_input",
]


def make_slice_and_unet_input(
    height: int,
    width: int,
    depth: int,
    pixel_size: float,
    center_xyz: Tuple[
        float, float, float
    ],  # (cx, cy, cz) in voxel coords (floats allowed)
    radius: float,  # physical units
    z_index: int = 0,
    clamp_image: bool = True,
    normalize: bool = True,
    normalize_radius: bool = False,  # NEW
    add_center_maps: bool = False,  # NEW
    norm_eps: float = 1e-12,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
      top_layer: (H, W) float32
      unet_x:    (C, H, W) float32 with base channels:
                 [dx, dy, dz, radius, pixel_size,
                  dist_left, dist_right, dist_top, dist_bottom,
                  W_phys, H_phys, D_phys,
                  x_phys, y_phys, z_phys]
                 plus optional appended channels if add_center_maps=True:
                  [+ cx_phys, cy_phys, cz_phys]
      L:         global physical length scale used for normalization (float)

    Normalization (if normalize=True):
      - Compute W_phys = width*pixel_size, H_phys = height*pixel_size, D_phys = depth*pixel_size
      - L = max(W_phys, H_phys, D_phys)
      - Divide all length-like channels by L:
          dx, dy, dz, pixel_size,
          dist_left/right/top/bottom,
          W_phys, H_phys, D_phys,
          x_phys, y_phys, z_phys,
          and center maps if enabled.
      - Radius channel:
          * if normalize_radius=False: kept in physical units (unnormalized)
          * if normalize_radius=True: radius/L (consistent with other length-like channels)
    """
    if height <= 0 or width <= 0 or depth <= 0:
        raise ValueError("height, width, depth must be positive integers.")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be > 0.")
    if radius < 0:
        raise ValueError("radius must be >= 0.")
    if not (0 <= z_index < depth):
        raise ValueError(f"z_index must be in [0, {depth-1}].")

    cx, cy, cz = center_xyz

    # Physical dimensions
    W_phys = float(width) * float(pixel_size)
    H_phys = float(height) * float(pixel_size)
    D_phys = float(depth) * float(pixel_size)

    # Global scale (physical units)
    L = float(max(W_phys, H_phys, D_phys))
    if not np.isfinite(L) or L <= 0:
        raise ValueError("Computed normalization scale L is invalid.")
    denom = max(L, norm_eps)

    # --- Coordinate grids (voxel coords) for this slice ---
    ys = np.arange(height, dtype=np.float64)[:, None]  # (H,1)
    xs = np.arange(width, dtype=np.float64)[None, :]  # (1,W)
    z = float(z_index)

    # --- Sphere top-layer image (linear falloff in physical units) ---
    dx_phys_1w = (xs - cx) * pixel_size  # (1,W)
    dy_phys_h1 = (ys - cy) * pixel_size  # (H,1)
    dz_phys = (z - cz) * pixel_size  # scalar

    r_phys = np.sqrt(
        dx_phys_1w * dx_phys_1w + dy_phys_h1 * dy_phys_h1 + dz_phys * dz_phys
    )

    if radius == 0:
        top_layer = np.zeros((height, width), dtype=dtype)
        xi = int(np.round(cx))
        yi = int(np.round(cy))
        if 0 <= xi < width and 0 <= yi < height and np.isclose(z, cz):
            top_layer[yi, xi] = 1.0
    else:
        top_layer = 1.0 - (r_phys / float(radius))
        if clamp_image:
            top_layer = np.clip(top_layer, 0.0, 1.0)
        top_layer = top_layer.astype(dtype)

    # --- UNet input channels (C,H,W) ---
    dx_map = np.broadcast_to(dx_phys_1w, (height, width)).astype(dtype)
    dy_map = np.broadcast_to(dy_phys_h1, (height, width)).astype(dtype)
    dz_map = np.full((height, width), dz_phys, dtype=dtype)

    radius_map = np.full((height, width), float(radius), dtype=dtype)
    px_map = np.full((height, width), float(pixel_size), dtype=dtype)

    # Boundary distance maps in physical units
    x_idx_1w = np.arange(width, dtype=np.float64)[None, :]  # (1,W)
    y_idx_h1 = np.arange(height, dtype=np.float64)[:, None]  # (H,1)

    dist_left = np.broadcast_to(x_idx_1w * pixel_size, (height, width)).astype(dtype)
    dist_right = np.broadcast_to(
        (width - 1 - x_idx_1w) * pixel_size, (height, width)
    ).astype(dtype)
    dist_top = np.broadcast_to(y_idx_h1 * pixel_size, (height, width)).astype(dtype)
    dist_bottom = np.broadcast_to(
        (height - 1 - y_idx_h1) * pixel_size, (height, width)
    ).astype(dtype)

    # Physical dims constant maps
    W_phys_map = np.full((height, width), W_phys, dtype=dtype)
    H_phys_map = np.full((height, width), H_phys, dtype=dtype)
    D_phys_map = np.full((height, width), D_phys, dtype=dtype)

    # Absolute position maps in physical units
    x_phys_map = dist_left
    y_phys_map = dist_top
    z_phys_map = np.full(
        (height, width), float(z_index) * float(pixel_size), dtype=dtype
    )

    # Optional center maps in physical units (constant maps)
    if add_center_maps:
        cx_phys = float(cx) * float(pixel_size)
        cy_phys = float(cy) * float(pixel_size)
        cz_phys = float(cz) * float(pixel_size)

        cx_phys_map = np.full((height, width), cx_phys, dtype=dtype)
        cy_phys_map = np.full((height, width), cy_phys, dtype=dtype)
        cz_phys_map = np.full((height, width), cz_phys, dtype=dtype)

    # --- Normalize (optional) ---
    if normalize:
        dx_map = dx_map / denom
        dy_map = dy_map / denom
        dz_map = dz_map / denom
        px_map = px_map / denom

        dist_left = dist_left / denom
        dist_right = dist_right / denom
        dist_top = dist_top / denom
        dist_bottom = dist_bottom / denom

        W_phys_map = W_phys_map / denom
        H_phys_map = H_phys_map / denom
        D_phys_map = D_phys_map / denom

        x_phys_map = x_phys_map / denom
        y_phys_map = y_phys_map / denom
        z_phys_map = z_phys_map / denom

        if add_center_maps:
            cx_phys_map = cx_phys_map / denom
            cy_phys_map = cy_phys_map / denom
            cz_phys_map = cz_phys_map / denom

        if normalize_radius:
            radius_map = radius_map / denom
    else:
        # Even if normalize=False, allow optional radius normalization relative to L for convenience
        if normalize_radius:
            radius_map = radius_map / denom
        # Center maps remain unnormalized if normalize=False

    # Stack base channels
    channels = [
        dx_map,
        dy_map,
        dz_map,
        radius_map,
        px_map,
        dist_left,
        dist_right,
        dist_top,
        dist_bottom,
        W_phys_map,
        H_phys_map,
        D_phys_map,
        x_phys_map,
        y_phys_map,
        z_phys_map,
    ]

    # Append center maps if requested
    if add_center_maps:
        channels.extend([cx_phys_map, cy_phys_map, cz_phys_map])

    unet_x = np.stack(channels, axis=0).astype(dtype)

    return top_layer, unet_x, L


# -------------------------
# Config sampler
# -------------------------


@dataclass(frozen=False)
class SphereSliceConfig:
    height: int
    width: int
    depth: int
    pixel_size: float
    center_xyz: Tuple[float, float, float]
    radius: float
    z_index: int


@dataclass
class SphereSliceRanges:
    height: Tuple[int, int] = (25, 100)
    width: Tuple[int, int] = (25, 100)
    depth: Tuple[int, int] = (3, 8)
    pixel_size: Tuple[float, float] = (0.5, 2.0)
    radius: Tuple[float, float] = (1.0, 50.0)
    z_index: int = 0

    # If you're training on crops, ensure sampled scenes are at least crop size:
    min_height: Optional[int] = None
    min_width: Optional[int] = None


def _rand_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def sample_config(
    rng: np.random.Generator, ranges, sampling: sphere_sampling.SamplingConfig
) -> tuple["SphereSliceConfig", dict]:
    # dimensions
    H = (
        sampling.fixed_h
        if sampling.fixed_hw
        else int(rng.integers(ranges.height[0], ranges.height[1] + 1))
    )
    W = (
        sampling.fixed_w
        if sampling.fixed_hw
        else int(rng.integers(ranges.width[0], ranges.width[1] + 1))
    )
    D = (
        sampling.fixed_depth
        if sampling.fixed_depth is not None
        else int(rng.integers(ranges.depth[0], ranges.depth[1] + 1))
    )
    pixel_size = (
        sampling.fixed_pixel_size
        if sampling.fixed_pixel_size is not None
        else float(rng.uniform(ranges.pixel_size[0], ranges.pixel_size[1]))
    )

    # radius stratification
    if sampling.use_radius_buckets:
        radius, r_bucket = sphere_sampling.sample_radius_bucketed(
            rng, sampling.radius_buckets
        )
    else:
        radius = float(rng.uniform(ranges.radius[0], ranges.radius[1]))
        r_bucket = -1

    # center
    if sampling.mode == "mixture":
        center_xyz, mode = sphere_sampling.sample_center_mixture(
            rng,
            H=H,
            W=W,
            D=D,
            pixel_size=float(pixel_size),
            radius=float(radius),
            z_index=int(ranges.z_index),
            cfg=sampling.mixture,
        )
    else:
        center_xyz = sphere_sampling.sample_center_uniform(rng, H, W, D)
        mode = "uniform"

    cfg = SphereSliceConfig(
        height=int(H),
        width=int(W),
        depth=int(D),
        pixel_size=float(pixel_size),
        center_xyz=center_xyz,
        radius=float(radius),
        z_index=int(ranges.z_index),
    )

    # meta for logging/analysis (no rendering required)
    cz = float(center_xyz[2])
    dz_vox = abs(cz - float(cfg.z_index))
    dz_phys = dz_vox * float(cfg.pixel_size)
    u = dz_phys / float(cfg.radius) if cfg.radius > 0 else 0.0
    r_slice = np.sqrt(max(cfg.radius**2 - dz_phys**2, 0.0))
    max_value = max(1.0 - u, 0.0)

    meta = {
        "radius_bucket": int(r_bucket),
        "mode": mode,
        "radius": float(cfg.radius),
        "cz": float(cz),
        "dz_phys": float(dz_phys),
        "u": float(u),
        "r_slice_phys": float(r_slice),
        "max_value": float(max_value),
    }
    return cfg, meta


# -------------------------
# Datasets
# -------------------------


class OnTheFlySphereSliceDataset(Dataset):
    """
    Generates samples on-the-fly. Deterministic per (epoch, index, worker) given base_seed,
    but changes when you call set_epoch(epoch).
    """

    def __init__(
        self,
        ranges: SphereSliceRanges,
        length: int,
        crop_size: Optional[Tuple[int, int]] = None,  # (crop_h, crop_w)
        base_seed: int = 12345,
        # forwarded to make_slice_and_unet_input:
        clamp_image: bool = True,
        normalize: bool = True,
        normalize_radius: bool = False,
        add_center_maps: bool = False,
        sampling: Optional[sphere_sampling.SamplingConfig] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.ranges = ranges
        self.length = int(length)
        self.crop_size = crop_size
        self.base_seed = int(base_seed)
        self.epoch = 0

        self.kw = dict(
            clamp_image=clamp_image,
            normalize=normalize,
            normalize_radius=normalize_radius,
            add_center_maps=add_center_maps,
        )
        self.dtype = dtype

        # If we crop, ensure sampled scenes are at least crop size
        if self.crop_size is not None:
            ch, cw = self.crop_size
            self.ranges.min_height = max(self.ranges.min_height or 0, ch)
            self.ranges.min_width = max(self.ranges.min_width or 0, cw)

        self.sampling = sampling or sphere_sampling.SamplingConfig()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def _make_rng(self, idx: int) -> np.random.Generator:
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0

        # Make a deterministic seed per (epoch, idx, worker).
        # This gives reproducibility and changes across epochs.
        seed = (self.base_seed + 1000003 * self.epoch + 9176 * worker_id + idx) % (
            2**32 - 1
        )
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int):
        rng = self._make_rng(idx)
        cfg, meta = sample_config(rng, self.ranges, sampling=self.sampling)

        # --- generate full slice + full feature map ---
        top_layer, unet_x, _L = make_slice_and_unet_input(
            height=cfg.height,
            width=cfg.width,
            depth=cfg.depth,
            pixel_size=cfg.pixel_size,
            center_xyz=cfg.center_xyz,
            radius=cfg.radius,
            z_index=cfg.z_index,
            **self.kw,
        )

        # Shapes:
        # top_layer: (H,W)
        # unet_x: (C,H,W)

        if self.crop_size is not None:
            ch, cw = self.crop_size
            H, W = top_layer.shape
            # Since we enforce H>=ch and W>=cw in sampling, this is safe:
            y0 = int(rng.integers(0, H - ch + 1))
            x0 = int(rng.integers(0, W - cw + 1))

            top_layer = top_layer[y0 : y0 + ch, x0 : x0 + cw]
            unet_x = unet_x[:, y0 : y0 + ch, x0 : x0 + cw]

        # Convert to torch
        x = torch.from_numpy(unet_x).to(self.dtype)  # (C,H,W)
        y = torch.from_numpy(top_layer).to(self.dtype).unsqueeze(0)  # (1,H,W)

        meta_t = {
            "radius_bucket": torch.tensor(meta["radius_bucket"], dtype=torch.long),
            "radius": torch.tensor(meta["radius"], dtype=torch.float32),
            "u": torch.tensor(meta["u"], dtype=torch.float32),
            "mode": meta["mode"],  # string is OK, kept as python object
        }

        return x, y, meta_t


class FixedSphereSliceDataset(Dataset):
    """
    Fixed validation set: sample N configs once using a seed, then generate deterministically.
    """

    def __init__(
        self,
        ranges: SphereSliceRanges,
        num_samples: int,
        crop_size: Optional[Tuple[int, int]] = None,
        seed: int = 999,
        clamp_image: bool = True,
        normalize: bool = True,
        normalize_radius: bool = False,
        add_center_maps: bool = False,
        sampling: Optional[sphere_sampling.SamplingConfig] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.ranges = ranges
        self.num_samples = int(num_samples)
        self.crop_size = crop_size
        self.seed = int(seed)

        self.kw = dict(
            clamp_image=clamp_image,
            normalize=normalize,
            normalize_radius=normalize_radius,
            add_center_maps=add_center_maps,
        )
        self.dtype = dtype

        self.configs: List[SphereSliceConfig] = []
        self.metas: List[dict] = []

        # Ensure sampled scenes are at least crop size
        if self.crop_size is not None:
            ch, cw = self.crop_size
            self.ranges.min_height = max(self.ranges.min_height or 0, ch)
            self.ranges.min_width = max(self.ranges.min_width or 0, cw)

        sampling = sampling or sphere_sampling.SamplingConfig()

        # Pre-sample configs (lightweight, no big arrays stored)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_samples):
            cfg, meta = sample_config(rng, self.ranges, sampling=sampling)
            self.configs.append(cfg)
            self.metas.append(meta)

        # Optional: for fixed crops per sample, pre-sample crop origins too (keeps val fully fixed)
        self.crop_origins: Optional[List[Tuple[int, int]]] = None
        if self.crop_size is not None:
            ch, cw = self.crop_size
            origins = []
            rng2 = np.random.default_rng(self.seed + 1234)
            for cfg in self.configs:
                # cfg.height/cfg.width are >= crop_size by construction
                y0 = int(rng2.integers(0, cfg.height - ch + 1))
                x0 = int(rng2.integers(0, cfg.width - cw + 1))
                origins.append((y0, x0))
            self.crop_origins = origins

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        cfg = self.configs[idx]
        meta = self.metas[idx]

        top_layer, unet_x, _L = make_slice_and_unet_input(
            height=cfg.height,
            width=cfg.width,
            depth=cfg.depth,
            pixel_size=cfg.pixel_size,
            center_xyz=cfg.center_xyz,
            radius=cfg.radius,
            z_index=cfg.z_index,
            **self.kw,
        )

        if self.crop_size is not None:
            ch, cw = self.crop_size
            y0, x0 = self.crop_origins[idx]  # fixed crop per sample
            top_layer = top_layer[y0 : y0 + ch, x0 : x0 + cw]
            unet_x = unet_x[:, y0 : y0 + ch, x0 : x0 + cw]

        x = torch.from_numpy(unet_x).to(self.dtype)
        y = torch.from_numpy(top_layer).to(self.dtype).unsqueeze(0)

        meta_t = {
            "radius_bucket": torch.tensor(meta["radius_bucket"], dtype=torch.long),
            "radius": torch.tensor(meta["radius"], dtype=torch.float32),
            "u": torch.tensor(meta["u"], dtype=torch.float32),
            "mode": meta["mode"],  # keep as string
        }

        return x, y, meta_t
