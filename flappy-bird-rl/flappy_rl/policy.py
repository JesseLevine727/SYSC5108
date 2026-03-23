from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[tuple[str, nn.Module]] = []
        input_dim = state_dim
        for index, hidden_dim in enumerate(hidden_dims):
            layers.append((f"linear_{index}", nn.Linear(input_dim, hidden_dim)))
            layers.append((f"tanh_{index}", nn.Tanh()))
            input_dim = hidden_dim

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.policy_head = nn.Linear(input_dim, action_dim)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


def infer_actor_critic_hidden_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    hidden_dims: list[int] = []
    index = 0
    while True:
        key = f"backbone.linear_{index}.weight"
        if key not in state_dict:
            break
        hidden_dims.append(int(state_dict[key].shape[0]))
        index += 1

    if hidden_dims:
        return tuple(hidden_dims)

    # Backward compatibility with older checkpoints that used numeric Sequential keys.
    legacy_hidden_dims: list[int] = []
    index = 0
    while True:
        key = f"backbone.{index}.weight"
        if key not in state_dict:
            break
        legacy_hidden_dims.append(int(state_dict[key].shape[0]))
        index += 2

    if legacy_hidden_dims:
        return tuple(legacy_hidden_dims)

    raise KeyError("Unable to infer ActorCritic hidden dimensions from checkpoint state_dict.")


def normalize_actor_critic_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "backbone.linear_0.weight" in state_dict:
        return state_dict

    normalized = dict(state_dict)
    index = 0
    legacy_key = 0
    while f"backbone.{legacy_key}.weight" in normalized:
        normalized[f"backbone.linear_{index}.weight"] = normalized.pop(f"backbone.{legacy_key}.weight")
        normalized[f"backbone.linear_{index}.bias"] = normalized.pop(f"backbone.{legacy_key}.bias")
        index += 1
        legacy_key += 2

    return normalized
