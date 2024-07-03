from typing import List

import torch
import torch.nn as nn

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        apply_batchnorm: bool = False,
        apply_dropout: bool = False,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        mlp_blocks = [
            mlp_block(
                in_features=in_f,
                out_features=out_f,
                apply_dropout=apply_dropout,
                dropout_rate=dropout_rate,
                apply_batchnorm=apply_batchnorm,
                apply_relu=True,
            )
            for in_f, out_f in zip([input_dim] + hidden_dims[:-1], hidden_dims)
        ]

        mlp_blocks.append(
            mlp_block(
                in_features=hidden_dims[-1],
                out_features=output_dim,
                apply_dropout=False,
                apply_batchnorm=False,
                apply_relu=False,
            )
        )

        self.mlp_blocks = nn.Sequential(*mlp_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_blocks(x)


def mlp_block(
    in_features: int,
    out_features: int,
    apply_batchnorm: bool,
    apply_relu: bool,
    apply_dropout: bool,
    dropout_rate: float = 0.5,
) -> nn.Module:
    # use nn.ModuleDict for each if-clause instead?
    mlp_blocks: List[nn.Module] = [
        nn.Linear(in_features=in_features, out_features=out_features)
    ]
    if apply_batchnorm:
        mlp_blocks.append(nn.BatchNorm1d(out_features))
    if apply_relu:
        mlp_blocks.append(nn.ReLU())
    if apply_dropout:
        mlp_blocks.append(nn.Dropout(p=dropout_rate))

    return nn.Sequential(*mlp_blocks)
