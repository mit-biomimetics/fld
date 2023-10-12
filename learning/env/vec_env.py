#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# python
from abc import ABC, abstractmethod
from typing import Tuple, Union

# torch
import torch


# minimal interface of the environment
class VecEnv(ABC):
    """Abstract class for vectorized environment."""

    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor  # current episode duration
    extras: dict
    device: torch.device

    """
    Properties
    """

    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass

    """
    Operations.
    """

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        """Apply input action on the environment.

        Args:
            actions (torch.Tensor): Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, dict]:
                A tuple containing the observations, privileged observations, rewards, dones and
                extra information (metrics).
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Reset all environment instances.

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None]: Tuple containing the observations and privileged observations.
        """
        raise NotImplementedError
