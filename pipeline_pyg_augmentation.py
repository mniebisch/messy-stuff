from typing import Optional

import torch
import torch_geometric.data as pyg_data
import torch_geometric.transforms as pyg_transforms
import torchdata


def geometric_augmentation(
    samples: torch.Tensor, transforms: Optional[pyg_transforms.BaseTransform] = None
):
    datapipe = torchdata.datapipes.iter.IterableWrapper(samples)
    datapipe = datapipe.map(create_geom_datapoint)
    datapipe = datapipe.batch(batch_size=5)
    if transforms is not None:
        datapipe = datapipe.map(transforms)
    datapipe = datapipe.map(unwrap_pyg_batch)
    return datapipe


def create_geom_datapoint(inputs):
    inputs = torch.reshape(inputs, (-1, 3))
    data = pyg_data.Data(pos=inputs)
    return data


def unwrap_pyg_batch(inputs):
    return torch.stack([torch.reshape(sample.pos, (-1,)) for sample in inputs])


if __name__ == "__main__":
    sample_data = torch.rand(20, 21 * 3, dtype=torch.float32)
    sample_data = torch.zeros((20, 21 * 3), dtype=torch.float32) + 1
    transforms = pyg_transforms.Compose(
        [
            pyg_transforms.NormalizeScale(),
            pyg_transforms.RandomFlip(axis=0, p=0.5),
            pyg_transforms.RandomJitter(0.01),
        ]
    )
    # transforms = pyg_transforms.Compose(
    #     [
    #         pyg_transforms.RandomRotate(degrees=25, axis=0),
    #         pyg_transforms.RandomRotate(degrees=25, axis=1),
    #         pyg_transforms.RandomRotate(degrees=25, axis=2),
    #     ]
    # )
    pipe = geometric_augmentation(samples=sample_data, transforms=transforms)
    for sample in pipe:
        print("blub")
