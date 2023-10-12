# isaacgym
from isaacgym import gymapi
from isaacgym import gymutil

# python
import sys
import torch
import abc
from typing import Tuple, Union

# solo-gym
from solo_gym.utils.base_config import BaseConfig


class BaseTask:
    """Base class for RL tasks."""

    def __init__(
        self,
        cfg: BaseConfig,
        sim_params: gymapi.SimParams,
        physics_engine: gymapi.SimType,
        sim_device: str,
        headless: bool,
    ):
        """Initialize the base class for RL.

        The class initializes the simulation. It also allocates buffers for observations,
        actions, rewards, reset, episode length, episode timetout and privileged observations.

        The :obj:`cfg` must contain the following:

        - num_envs (int): Number of environment instances.
        - num_observations (int): Number of observations.
        - num_privileged_obs (int): Number of privileged observations.
        - num_actions (int): Number of actions.

        Note:
            If :obj:`cfg.num_privileged_obs` is not :obj:`None`, a buffer for privileged
            observations is returned. This is useful for critic observations in asymmetric
            actor-critic.

        Args:
            cfg (BaseConfig): Configuration for the environment.
            sim_params (gymapi.SimParams): The simulation parameters.
            physics_engine (gymapi.SimType): Simulation type (must be gymapi.SIM_PHYSX).
            sim_device (str): The simulation device (ex: `cuda:0` or `cpu`).
            headless (bool): If true, run without rendering.
        """
        # copy input arguments into class members
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.headless = headless
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        # env device is GPU only if sim is on GPU and use_gpu_pipeline is True.
        # otherwise returned tensors are copied to CPU by PhysX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"
        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless is True:
            self.graphics_device_id = -1

        # store the environment information
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs,
                self.num_privileged_obs,
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.privileged_obs_buf = None
        # allocate dictionary to store metrics
        self.extras = {}

        # create envs, sim
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # create viewer
        # Todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless is False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def __del__(self):
        """Cleanup in the end."""
        try:
            if self.sim is not None:
                self.gym.destroy_sim(self.sim)
            if self.viewer is not None:
                self.gym.destroy_viewer(self.viewer)
        except:
            pass

    """
    Properties.
    """

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        return self.privileged_obs_buf

    """
    Operations.
    """

    def set_camera_view(self, position: Tuple[float, float, float], lookat: Tuple[float, float, float]) -> None:
        """Set camera position and direction."""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def reset(self) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Reset all environment instances.

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None]: Tuple containing the observations and privileged observations.
        """
        # reset environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # perform single-step to get observations
        zero_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        obs, privileged_obs, _, _, _ = self.step(zero_actions)
        # return obs
        return obs, privileged_obs

    @abc.abstractmethod
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

    def render(self, sync_frame_time=True):
        """Render the viewer."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    """
    Protected Methods.
    """

    @abc.abstractmethod
    def create_sim(self):
        """Creates simulation, terrain and environments"""
        raise NotImplementedError

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Resets the MDP for given environment instances.

        Args:
            env_ids (torch.Tensor): A tensor containing indices of environment instances to reset.
        """
        raise NotImplementedError
