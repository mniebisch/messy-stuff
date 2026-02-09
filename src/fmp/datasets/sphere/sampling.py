from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "MixtureSamplerConfig",
    "SamplingConfig",
    "SoftDzSamplerConfig",
    "sample_center_uniform",
    "sample_center_mixture",
]


@dataclass
class MixtureSamplerConfig:
    # probabilities
    p_strong: float = 0.7
    p_medium: float = 0.20
    p_faint: float = 0.2
    p_zero: float = 0.1

    # regime parameters
    strong_alpha: float = 0.4  # strong: dz <= alpha*r
    medium_lo: float = 0.4  # medium: dz in [medium_lo*r, medium_hi*r]
    medium_hi: float = 0.8
    faint_beta: float = 0.8  # faint: dz in [beta*r, (1-eps)*r]
    zero_gamma: float = 0.1  # zero: dz >= (1+gamma)*r

    eps: float = 1e-3


@dataclass
class SoftDzSamplerConfig:
    """
    Sample u = dz/r from a mixture of uniform intervals on [0, 1],
    plus an optional zero-component sampled from [u_zero_min, +inf) but clipped by cube depth.
    """

    # components on [0,1] (each is [lo,hi] with a weight)
    components: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.0, 0.4, 0.55),  # strong-ish
            (0.4, 0.8, 0.20),  # medium
            (0.8, 1.0, 0.15),  # faint
        ]
    )

    # optional zero sampling
    p_zero: float = 0.10
    u_zero_min: float = 1.1  # dz/r >= 1.1 => guaranteed zero (if feasible)

    eps: float = 1e-3


@dataclass
class SamplingConfig:
    """
    mode:
      - "uniform": sample center uniformly in the cube (your current behavior)
      - "mixture": sample cz conditioned on radius and z_index (strong/faint/zero mixture)
      - "soft": sample dz/r from a mixture of uniform intervals, plus an optional zero component
    """

    mode: str = "uniform"  # "uniform" | "mixture" | "soft"
    mixture: MixtureSamplerConfig = field(default_factory=MixtureSamplerConfig)
    soft_dz_mixture: SoftDzSamplerConfig = field(default_factory=SoftDzSamplerConfig)

    # optional stabilization knobs
    fixed_hw: bool = False
    fixed_h: int = 100
    fixed_w: int = 100
    fixed_depth: Optional[int] = None  # e.g. 8
    fixed_pixel_size: Optional[float] = None  # e.g. 1.0


def _clip(
    lo: float, hi: float, b_lo: float, b_hi: float
) -> Optional[Tuple[float, float]]:
    lo2 = max(lo, b_lo)
    hi2 = min(hi, b_hi)
    if lo2 > hi2:
        return None
    return (lo2, hi2)


def _sample_from_intervals(
    rng: np.random.Generator, intervals: list[Tuple[float, float]]
) -> float:
    lengths = np.array([hi - lo for lo, hi in intervals], dtype=np.float64)
    probs = lengths / lengths.sum()
    idx = int(rng.choice(len(intervals), p=probs))
    lo, hi = intervals[idx]
    return float(rng.uniform(lo, hi))


def sample_center_uniform(
    rng: np.random.Generator, H: int, W: int, D: int
) -> Tuple[float, float, float]:
    cx = float(rng.uniform(0.0, max(0.0, W - 1.0)))
    cy = float(rng.uniform(0.0, max(0.0, H - 1.0)))
    cz = float(rng.uniform(0.0, max(0.0, D - 1.0)))
    return cx, cy, cz


def _sample_cz_from_dz(
    rng: np.random.Generator, z0: float, dz: float, cz_min: float, cz_max: float
) -> Optional[float]:
    """
    Sample cz such that |cz - z0| is approximately dz, by choosing one of the two sides.
    """
    intervals = []
    left = _clip(z0 - dz, z0 - dz, cz_min, cz_max)  # single point (still ok)
    right = _clip(z0 + dz, z0 + dz, cz_min, cz_max)
    # single point intervals have 0 length, so instead expand by tiny epsilon:
    eps = 1e-6
    left = _clip(z0 - dz - eps, z0 - dz + eps, cz_min, cz_max)
    right = _clip(z0 + dz - eps, z0 + dz + eps, cz_min, cz_max)
    if left is not None:
        intervals.append(left)
    if right is not None:
        intervals.append(right)
    if not intervals:
        return None
    return _sample_from_intervals(rng, intervals)


def sample_center_mixture(
    rng: np.random.Generator,
    *,
    H: int,
    W: int,
    D: int,
    pixel_size: float,
    radius: float,
    z_index: int,
    cfg: MixtureSamplerConfig,
) -> Tuple[float, float, float]:
    cx, cy, _ = sample_center_uniform(rng, H, W, D)

    z0 = float(z_index)
    cz_min, cz_max = 0.0, float(D - 1)
    r_vox = radius / float(pixel_size)

    probs = np.array(
        [cfg.p_strong, cfg.p_medium, cfg.p_faint, cfg.p_zero], dtype=np.float64
    )
    probs = probs / probs.sum()
    mode = str(rng.choice(["strong", "medium", "faint", "zero"], p=probs))

    def try_mode(m: str) -> Optional[float]:
        if m == "strong":
            dz_max = cfg.strong_alpha * r_vox
            interval = _clip(z0 - dz_max, z0 + dz_max, cz_min, cz_max)
            if interval is None:
                return None
            return float(rng.uniform(interval[0], interval[1]))

        if m == "medium":
            dz_lo = cfg.medium_lo * r_vox
            dz_hi = cfg.medium_hi * r_vox
            if dz_lo > dz_hi:
                return None
            left = _clip(z0 - dz_hi, z0 - dz_lo, cz_min, cz_max)
            right = _clip(z0 + dz_lo, z0 + dz_hi, cz_min, cz_max)
            intervals = [x for x in (left, right) if x is not None]
            if not intervals:
                return None
            return _sample_from_intervals(rng, intervals)

        if m == "faint":
            dz_lo = cfg.faint_beta * r_vox
            dz_hi = (1.0 - cfg.eps) * r_vox
            if dz_lo > dz_hi:
                return None
            left = _clip(z0 - dz_hi, z0 - dz_lo, cz_min, cz_max)
            right = _clip(z0 + dz_lo, z0 + dz_hi, cz_min, cz_max)
            intervals = [x for x in (left, right) if x is not None]
            if not intervals:
                return None
            return _sample_from_intervals(rng, intervals)

        if m == "zero":
            dz_min = (1.0 + cfg.zero_gamma) * r_vox
            left = _clip(cz_min, z0 - dz_min, cz_min, cz_max)
            right = _clip(z0 + dz_min, cz_max, cz_min, cz_max)
            intervals = [x for x in (left, right) if x is not None]
            if not intervals:
                return None
            return _sample_from_intervals(rng, intervals)

        raise ValueError(m)

    # chosen mode, then fallbacks
    for m in [mode, "strong", "medium", "faint", "zero"]:
        cz = try_mode(m)
        if cz is not None:
            return cx, cy, cz

    # fallback to uniform
    _, _, cz = sample_center_uniform(rng, H, W, D)
    return cx, cy, cz


def sample_center_soft(
    rng: np.random.Generator,
    *,
    H: int,
    W: int,
    D: int,
    pixel_size: float,
    radius: float,
    z_index: int,
    cfg: SoftDzSamplerConfig,
) -> Tuple[float, float, float]:
    """
    Soft selection: sample u = dz/r from a mixture over [0,1], then choose cz accordingly.
    Also supports a "zero" component with u >= u_zero_min (if feasible).
    """
    cx, cy, _ = sample_center_uniform(rng, H, W, D)

    z0 = float(z_index)
    cz_min, cz_max = 0.0, float(D - 1)
    r_vox = radius / float(pixel_size)

    # Build component weights for u in [0,1]
    comps = list(cfg.components)
    # normalize comp weights
    w = np.array([c[2] for c in comps], dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum <= 0:
        raise ValueError("SoftDzSamplerConfig.components weights must sum > 0")
    w = w / w_sum

    # Decide if we sample "zero" or in-sphere u
    if cfg.p_zero > 0 and rng.random() < cfg.p_zero:
        # sample dz >= u_zero_min * r_vox (clipped by cube)
        dz_min = cfg.u_zero_min * r_vox
        # feasible cz intervals for |cz-z0| >= dz_min
        left = _clip(cz_min, z0 - dz_min, cz_min, cz_max)
        right = _clip(z0 + dz_min, cz_max, cz_min, cz_max)
        intervals = [x for x in (left, right) if x is not None]
        if intervals:
            cz = _sample_from_intervals(rng, intervals)
            return cx, cy, cz
        # if impossible, fall back to in-sphere sampling below

    # sample u from mixture of intervals on [0,1]
    idx = int(rng.choice(len(comps), p=w))
    lo, hi, _ = comps[idx]
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))
    if lo > hi:
        lo, hi = hi, lo
    # keep away from exactly 1.0 by eps (intersection boundary)
    hi = min(hi, 1.0 - cfg.eps)
    u = float(rng.uniform(lo, hi))
    dz = u * r_vox

    # sample cz from either side at distance dz
    cz = _sample_cz_from_dz(rng, z0=z0, dz=dz, cz_min=cz_min, cz_max=cz_max)
    if cz is not None:
        return cx, cy, cz

    # fallback: uniform cz
    _, _, cz = sample_center_uniform(rng, H, W, D)
    return cx, cy, cz
