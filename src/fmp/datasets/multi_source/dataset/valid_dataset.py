from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ["ImageRowDataset"]


class ImageRowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        source_name: str,
        source_id: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        image_loader: Optional[Callable[[str], torch.Tensor]] = None,
        target_col: Optional[Union[str, Sequence[str]]] = None,
        target_transform: Optional[Callable[[Any], torch.Tensor]] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.source_name = source_name
        self.source_id = int(source_id)
        self.transform = transform
        self.image_loader = image_loader or self._default_image_loader
        self.target_col = target_col
        self.target_transform = target_transform

        if "file_path" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'file_path' column.")

        if self.target_col is not None:
            cols = (
                [self.target_col]
                if isinstance(self.target_col, str)
                else list(self.target_col)
            )
            missing = [c for c in cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"Missing target columns in df: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        path = row["file_path"]

        image = self.image_loader(path)  # Tensor [C,H,W] recommended
        if self.transform is not None:
            image = self.transform(image)

        target = None
        if self.target_col is not None:
            if isinstance(self.target_col, str):
                raw_t = row[self.target_col]
            else:
                raw_t = row[list(self.target_col)].to_numpy()

            if self.target_transform is not None:
                target = self.target_transform(raw_t)
            else:
                # sensible default: float tensor for numeric targets
                target = torch.as_tensor(raw_t)

        return {
            "image": image,
            "target": target,
            "source": self.source_name,
            "source_id": self.source_id,
            "file_path": path,
        }

    @staticmethod
    def _default_image_loader(path: str) -> torch.Tensor:
        raise NotImplementedError(
            "Provide an image_loader that returns Tensor [C,H,W]."
        )
