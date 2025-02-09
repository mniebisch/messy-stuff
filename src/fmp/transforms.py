from typing import Tuple, Union

import torch
from torchvision.transforms import v2

__all__ = ["PadToSize"]


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
