from typing import List, Optional

import torch
import torch_geometric.data as pyg_data
import torch_geometric.transforms as pyg_transforms
import torchdata


def create_geom_datapoint(inputs: torch.Tensor) -> pyg_data.Data:
    if inputs.shape != (63,) and inputs.dim() != 1:
        raise ValueError("Function is not intended to be applied to batched data")
    inputs = torch.reshape(inputs, (-1, 3))
    data = pyg_data.Data(pos=inputs)
    return data


def unwrap_pyg_datapoint(inputs: pyg_data.Data) -> torch.Tensor:
    return torch.reshape(inputs.pos, (-1,))


def collate_pyg_datapoint(batch: List[pyg_data.Data]) -> torch.Tensor:
    return torch.stack([unwrap_pyg_datapoint(sample) for sample in batch])


if __name__ == "__main__":
    # TODO how are pyg transforms applied? does the current code lead to a speed up?

    def create_prototyping_pipe(
        samples: torch.Tensor,
        batch_size: int,
        drop_last_batch: bool = True,
        transforms: Optional[pyg_transforms.BaseTransform] = None,
    ) -> torchdata.datapipes.iter.IterDataPipe:
        datapipe = torchdata.datapipes.iter.IterableWrapper(samples)

        if transforms is not None:
            datapipe = datapipe.map(create_geom_datapoint)

        if transforms is not None:
            datapipe = datapipe.map(transforms)

        datapipe = datapipe.batch(batch_size=batch_size, drop_last=drop_last_batch)

        # if transforms is not None:
        #     datapipe = datapipe.map(transforms)

        if transforms is not None:
            datapipe = datapipe.collate(collate_fn=collate_pyg_datapoint)
        else:
            datapipe = datapipe.collate()
        return datapipe

    sample_data = torch.rand(20, 21 * 3, dtype=torch.float32)
    sample_data = torch.zeros((10000, 21 * 3), dtype=torch.float32) + 1
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
    pipe = create_prototyping_pipe(
        samples=sample_data, batch_size=64, drop_last_batch=False, transforms=transforms
    )
    for sample in pipe:
        print("blub")
