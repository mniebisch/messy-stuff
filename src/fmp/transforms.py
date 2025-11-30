import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import scale
from numpy import typing as npt
from PIL import Image
from scipy.signal import convolve2d
from torchvision import tv_tensors
from torchvision.transforms import v2

__all__ = [
    "BackgroundImage",
    "CropOrPad",
    "ImageSharpening",
    "FactorResize",
    "FXAALite",
    "PadToSize",
    "ScaleJitter",
    "ScaleJitterXY",
]


class PadToSize(v2.Pad):
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        fill: int = 0,
        padding_mode: str = "constant",
    ):
        super().__init__(padding=0, fill=fill, padding_mode=padding_mode)
        self.size = size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if isinstance(self.size, int):
            target_height, target_width = self.size, self.size
        else:
            target_height, target_width = self.size

        height, width = v2.query_size(img)

        if height == target_height and width == target_width:
            return img
        elif height > target_height or width > target_width:
            raise ValueError(
                f"Image size ({height}, {width}) is larger than target size ({target_height}, {target_width})"
            )

        padding_height = target_height - height
        padding_width = target_width - width

        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        return v2.functional.pad(
            img,
            (padding_left, padding_top, padding_right, padding_bottom),
            fill=self.fill,
            padding_mode=self.padding_mode,
        )


class ScaleJitter(v2.ScaleJitter):
    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_height, orig_width = v2.query_size(flat_inputs)

        scale = self.scale_range[0] + torch.rand(1) * (
            self.scale_range[1] - self.scale_range[0]
        )
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        return dict(size=(new_height, new_width))


class ScaleJitterXY(v2.ScaleJitter):
    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_height, orig_width = v2.query_size(flat_inputs)

        random_value = torch.rand(2)
        scale_x = self.scale_range[0] + random_value[0] * (
            self.scale_range[1] - self.scale_range[0]
        )
        scale_y = self.scale_range[0] + random_value[1] * (
            self.scale_range[1] - self.scale_range[0]
        )

        new_width = int(orig_width * scale_x)
        new_height = int(orig_height * scale_y)

        return dict(size=(new_height, new_width))


class CropOrPad(torch.nn.Module):
    def __init__(
        self,
        target_size: Tuple[int, int],
        fill: int = 0,
        padding_mode: str = "constant",
        random_crop: bool = False,
    ):
        super().__init__()
        self.target_size = target_size
        self.fill = fill
        self.padding_mode = padding_mode
        self.random_crop = random_crop

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        target_h, target_w = self.target_size
        # Assume image shape is (C, H, W)
        orig_h, orig_w = v2.query_size(img)

        # --- Adjust Height ---
        if orig_h > target_h:
            if self.random_crop:
                crop_top = random.randint(0, orig_h - target_h)
            else:
                crop_top = (orig_h - target_h) // 2
            img = img[:, crop_top : crop_top + target_h, :]
        elif orig_h < target_h:
            pad_total = target_h - orig_h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            # Padding order: (left, top, right, bottom)
            img = v2.functional.pad(
                img,
                (0, pad_top, 0, pad_bottom),
                fill=self.fill,
                padding_mode=self.padding_mode,
            )

        # --- Adjust Width ---
        if orig_w > target_w:
            if self.random_crop:
                crop_left = random.randint(0, orig_w - target_w)
            else:
                crop_left = (orig_w - target_w) // 2
            img = img[:, :, crop_left : crop_left + target_w]
        elif orig_w < target_w:
            pad_total = target_w - orig_w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            # Padding order: (left, top, right, bottom)
            img = v2.functional.pad(
                img,
                (pad_left, 0, pad_right, 0),
                fill=self.fill,
                padding_mode=self.padding_mode,
            )

        return img


class BackgroundImage(torch.nn.Module):
    def __init__(
        self,
        data_root: str,
        image_transforms: Optional[v2.Transform] = None,
        pad_fill: int = 0,
        pad_padding_mode: str = "constant",
    ) -> None:
        super().__init__()
        classes = [
            "bedroom_train",
            "classroom_train",
            "conference_room_train",
            "dining_room_train",
            "kitchen_train",
            "living_room_train",
        ]

        self.fill = pad_fill
        self.padding_mode = pad_padding_mode
        self.padding_mode = "constant"
        self.background_data = torchvision.datasets.LSUN(
            root=data_root, classes=classes, transform=image_transforms
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        background_image = self.background_data[
            random.randint(0, len(self.background_data))
        ][0]
        bg_h, bg_w = v2.query_size(background_image)
        h, w = v2.query_size(img)
        new_h = max(bg_h, h)
        new_w = max(bg_w, w)
        pad_top = (new_h - bg_h) // 2
        pad_bottom = new_h - bg_h - pad_top
        pad_left = (new_w - bg_w) // 2
        pad_right = new_w - bg_w - pad_left
        composite = v2.functional.pad(
            background_image,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
            padding_mode=self.padding_mode,
        )
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        composite[..., top : top + h, left : left + w] = img
        return composite


class FactorResize(torch.nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h, w = v2.query_size(img)
        new_h = int(h * self.factor)
        new_w = int(w * self.factor)
        return v2.functional.resize(
            img,
            (new_h, new_w),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )


class FXAALite(v2.Transform):
    """FXAA-lite anti-aliasing as a torchvision v2 transform.

    Accepts ``tv_tensors.Image`` (or PIL / NumPy), returns ``tv_tensors.Image`` **RGB, uint8**.
    """

    def __init__(
        self, tau: float = 12, dilate_k: int = 5, sigma: float = 0.9, passes: int = 2
    ) -> None:
        super().__init__()
        self.tau = tau
        self.dilate_k = dilate_k
        self.sigma = sigma
        self.passes = passes

    def transform(self, inpt: Any, param: Dict[str, Any]) -> tv_tensors.Image:
        # -- convert to a NumPy uint8 RGB array —
        if isinstance(inpt, tv_tensors.Image):
            arr_rgb = np.array(inpt)  # HWC, uint8
        elif isinstance(inpt, Image.Image):
            arr_rgb = np.array(inpt)
        elif isinstance(inpt, np.ndarray):
            arr_rgb = inpt
            if arr_rgb.dtype != np.uint8:
                raise ValueError("NumPy input must be uint8 [0-255]")
        else:
            raise TypeError(f"Unsupported input type: {type(inpt)}")

        # -- BGR-in / BGR-out for OpenCV
        arr_rgb = np.transpose(arr_rgb, (1, 2, 0))
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        out_bgr = fxaa_antialias_bgr(
            arr_bgr,
            tau=self.tau,
            dilate_k=self.dilate_k,
            sigma=self.sigma,
            passes=self.passes,
        )
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        out_rgb = np.transpose(out_rgb, (2, 0, 1))

        return tv_tensors.Image(out_rgb)


def fxaa_antialias_bgr(
    img_bgr: np.ndarray,
    tau: float = 12,
    dilate_k: int = 5,
    sigma: float = 0.9,
    passes: int = 2,
) -> np.ndarray:
    """FXAA-lite anti-aliasing (OpenCV + NumPy).  img_bgr must be uint8 BGR."""
    imgf = img_bgr.astype(np.float32) / 255.0
    for _ in range(max(1, passes)):
        # 1) detect high-frequency edges via Laplacian
        gray = cv2.cvtColor((imgf * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        mask = (np.abs(lap) > tau).astype(np.float32)

        # 2) enlarge mask so blur covers entire stair pattern
        mask = cv2.dilate(mask, np.ones((dilate_k, dilate_k), np.float32), iterations=1)

        # 3) local blur on masked pixels only
        blur = cv2.GaussianBlur(imgf, (0, 0), sigmaX=sigma, sigmaY=sigma)
        imgf = blur * mask[..., None] + imgf * (1.0 - mask[..., None])

    # back to uint8 BGR
    return (imgf * 255.0).clip(0, 255).astype(np.uint8)


# ?https://kornia.readthedocs.io/en/latest/enhance.html#kornia.enhance.sharpness


class ImageSharpening(v2.Transform):

    def __init__(self) -> None:
        super().__init__()

    def transform(self, inpt: Any, param: Dict[str, Any]) -> tv_tensors.Image:
        # -- convert to a NumPy uint8 RGB array —
        if isinstance(inpt, tv_tensors.Image):
            arr_rgb = np.array(inpt)  # HWC, uint8
        elif isinstance(inpt, Image.Image):
            arr_rgb = np.array(inpt)
        elif isinstance(inpt, np.ndarray):
            arr_rgb = inpt
            if arr_rgb.dtype != np.uint8:
                raise ValueError("NumPy input must be uint8 [0-255]")
        else:
            raise TypeError(f"Unsupported input type: {type(inpt)}")

        arr_rgb = np.transpose(arr_rgb, (1, 2, 0))

        output = image_sharpening(arr_rgb)

        output = np.transpose(output, (2, 0, 1))

        return tv_tensors.Image(output)


def image_sharpening(image: npt.NDArray) -> npt.NDArray:
    """Apply a sharpening filter to the image."""
    image = image.astype(np.float32)
    laplacian = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    image_laplacian = apply_kernel(image, laplacian)
    sobel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    image_sobel_vertical = apply_kernel(image, sobel_vertical)
    image_sobel_horizontal = apply_kernel(image, sobel_horizontal)
    image_sobel = np.abs(image_sobel_vertical) + np.abs(image_sobel_horizontal)
    box_kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    image_sobel = apply_kernel(image_sobel, box_kernel)

    return np.clip(image - image_laplacian * (image_sobel / 255), 0, 255).astype(
        np.uint8
    )


def apply_kernel(image: npt.NDArray, kernel: npt.NDArray) -> npt.NDArray:
    """Apply a convolution kernel to the image."""
    if image.ndim == 2:  # Grayscale image
        return convolve2d(image, kernel, mode="same", boundary="wrap")
    elif image.ndim == 3:  # Color image
        return np.stack(
            [
                convolve2d(
                    image[:, :, channel_index], kernel, mode="same", boundary="symm"
                )
                for channel_index in range(image.shape[2])
            ],
            axis=-1,
        )
    else:
        raise ValueError("Unsupported image shape. Expected 2D or 3D array.")
