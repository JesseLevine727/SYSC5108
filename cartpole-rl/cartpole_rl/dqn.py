from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[tuple[str, nn.Module]] = []
        input_dim = state_dim
        for index, hidden_dim in enumerate(hidden_dims):
            layers.append((f"linear_{index}", nn.Linear(input_dim, hidden_dim)))
            layers.append((f"relu_{index}", nn.ReLU()))
            input_dim = hidden_dim
        layers.append(("output", nn.Linear(input_dim, action_dim)))
        self.network = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def infer_q_hidden_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    hidden_dims: list[int] = []
    index = 0
    while True:
        key = f"network.linear_{index}.weight"
        if key not in state_dict:
            break
        hidden_dims.append(int(state_dict[key].shape[0]))
        index += 1
    if not hidden_dims:
        raise KeyError("Unable to infer QNetwork hidden dimensions from checkpoint state_dict.")
    return tuple(hidden_dims)
