import random  # added for random cropping
from typing import Tuple, Union

import torch
from torchvision.transforms import v2

__all__ = ["PadToSize", "ScaleJitter"]


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

        height, width = img.shape[-2:]

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


class ScaleJitter(torch.nn.Module):
    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float],
        fill: int = 0,
        padding_mode: str = "constant",
        random_crop: bool = False,
    ):
        """
        Args:
            target_size: Desired output size as (height, width)
            scale_range: Tuple (min_scale, max_scale) for random scaling
            fill: Fill value for padding (default=0)
            padding_mode: Padding mode passed to v2.functional.pad (default="constant")
            random_crop: Flag to enable random cropping (default=False)
        """
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.fill = fill
        self.padding_mode = padding_mode
        self.random_crop = random_crop

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Apply random scale jitter using torchvision's transform
        scaler = v2.ScaleJitter(self.target_size, self.scale_range)
        scaled_image = scaler(img)
        target_h, target_w = self.target_size
        # Assume image shape is (C, H, W)
        _, orig_h, orig_w = scaled_image.shape

        # --- Adjust Height ---
        if orig_h > target_h:
            if self.random_crop:
                crop_top = random.randint(0, orig_h - target_h)
            else:
                crop_top = (orig_h - target_h) // 2
            scaled_image = scaled_image[:, crop_top : crop_top + target_h, :]
        elif orig_h < target_h:
            pad_total = target_h - orig_h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            # Padding order: (left, top, right, bottom)
            scaled_image = v2.functional.pad(
                scaled_image,
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
            scaled_image = scaled_image[:, :, crop_left : crop_left + target_w]
        elif orig_w < target_w:
            pad_total = target_w - orig_w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            # Padding order: (left, top, right, bottom)
            scaled_image = v2.functional.pad(
                scaled_image,
                (pad_left, 0, pad_right, 0),
                fill=self.fill,
                padding_mode=self.padding_mode,
            )

        return scaled_image
