from typing import Any, TypedDict

import torch


class State(TypedDict):
    """The training state."""

    epoch: int
    model_state_dict: dict[str, torch.Tensor]
    optim_state_dict: dict[str, Any]
    loss: float
    all_losses: list[float]
