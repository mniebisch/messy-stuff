from typing import Optional, Union

import torch
import torch_geometric.data as pyg_data
import torch_geometric.transforms as pyg_transforms
import torchdata


def geometric_augmentation(
    datapipe: Union[
        torchdata.datapipes.iter.IterDataPipe, torchdata.datapipes.map.MapDataPipe
    ],
    transforms: pyg_transforms.BaseTransform,
) -> torchdata.datapipes.map.MapDataPipe:
    # TODO at the moment function needs to be applied before batching
    # make more generic
    # TODO if unbatched version is applied maybe internally apply batching anyway
    # to save computational cost? (be aware  of premature optimization!)
    datapipe = datapipe.map(create_geom_datapoint)
    datapipe = datapipe.map(transforms)
    datapipe = datapipe.map(unwrap_pyg_datapoint)
    return datapipe


def create_geom_datapoint(inputs):
    if inputs.shape != (63,) and inputs.dim() != 1:
        raise ValueError("Function is not intended to be applied to batched data")
    inputs = torch.reshape(inputs, (-1, 3))
    data = pyg_data.Data(pos=inputs)
    return data


def unwrap_pyg_datapoint(inputs):
    return torch.reshape(inputs.pos, (-1,))


if __name__ == "__main__":

    def create_prototyping_pipe(
        samples: torch.Tensor,
        batch_size: int,
        drop_last_batch: bool = True,
        transforms: Optional[pyg_transforms.BaseTransform] = None,
    ) -> torchdata.datapipes.iter.IterDataPipe:
        datapipe = torchdata.datapipes.iter.IterableWrapper(samples)
        if transforms is not None:
            datapipe = geometric_augmentation(datapipe=datapipe, transforms=transforms)
        datapipe = datapipe.batch(batch_size=batch_size, drop_last=drop_last_batch)
        datapipe = datapipe.collate()
        return datapipe

    sample_data = torch.rand(20, 21 * 3, dtype=torch.float32)
    sample_data = torch.zeros((20, 21 * 3), dtype=torch.float32) + 1
    transforms = pyg_transforms.Compose(
        [
            # pyg_transforms.NormalizeScale(),
            pyg_transforms.RandomFlip(axis=0, p=0.5),
            # pyg_transforms.RandomJitter(0.01),
        ]
    )
    # transforms = pyg_transforms.Compose(
    #     [
    #         pyg_transforms.RandomRotate(degrees=25, axis=0),
    #         pyg_transforms.RandomRotate(degrees=25, axis=1),
    #         pyg_transforms.RandomRotate(degrees=25, axis=2),
    #     ]
    # )
    # pipe = geometric_augmentation(samples=sample_data, transforms=transforms)
    pipe = create_prototyping_pipe(
        samples=sample_data, batch_size=5, drop_last_batch=False, transforms=transforms
    )
    for sample in pipe:
        print("blub")
