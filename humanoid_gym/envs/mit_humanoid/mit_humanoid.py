# python
import torch

# legged-gym
from humanoid_gym.envs import LeggedRobot
from .mit_humanoid_config import MITHumanoidFlatCfg
from isaacgym import gymapi
from isaacgym.torch_utils import (
    torch_rand_float,
    quat_rotate_inverse,
)
from learning.modules.fld import FLD
from typing import Dict
from humanoid_gym.utils.keyboard_controller import KeyboardAction, Delta, Switch

class MITHumanoid(LeggedRobot):
    cfg: MITHumanoidFlatCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.fld_latent_channel = self.cfg.fld.latent_channel
        self.fld_observation_horizon = self.cfg.fld.observation_horizon
        self.fld_state_idx_dict = self.cfg.fld.state_idx_dict
        self.fld_dim_of_interest = torch.cat([torch.tensor(ids, device=self.device, dtype=torch.long, requires_grad=False) for state, ids in self.fld_state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
        self.fld_observation_dim = len(self.fld_dim_of_interest)
        self.fld_observation_buf = torch.zeros(
            self.num_envs, self.fld_observation_horizon, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.fld_state = torch.zeros(self.num_envs, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)

        self.fld = FLD(self.fld_observation_dim, self.fld_observation_horizon, self.fld_latent_channel, self.device, encoder_shape=self.cfg.fld.encoder_shape, decoder_shape=self.cfg.fld.decoder_shape).eval()
        fld_load_root = self.cfg.fld.load_root
        if fld_load_root is not None:
            fld_load_model = self.cfg.fld.load_model
            loaded_dict = torch.load(fld_load_root + "/" + fld_load_model)
            self.fld.load_state_dict(loaded_dict["fld_state_dict"])
            self.fld.eval()
            statistics_dict = torch.load(fld_load_root + "/statistics.pt")
            self.state_transitions_mean, self.state_transitions_std = statistics_dict["state_transitions_mean"], statistics_dict["state_transitions_std"]
            self.latent_param_max, self.latent_param_min, self.latent_param_mean, self.latent_param_std = statistics_dict["latent_param_max"], statistics_dict["latent_param_min"], statistics_dict["latent_param_mean"], statistics_dict["latent_param_std"]
        self.decoded_obs = torch.zeros(self.num_envs, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.decoded_obs_state_idx_dict = {}
        current_length = 0
        for state, ids in self.fld_state_idx_dict.items():
            if (state != "base_pos") and (state != "base_quat"):
                self.decoded_obs_state_idx_dict[state] = list(range(current_length, current_length + len(ids)))
                current_length = current_length + len(ids)
        self.task_sampler = None

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg.asset.enable_joint_force_sensors:
            self.gym.refresh_dof_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.update_fld_observation_buf()
        self.get_latent_params()
        self.compute_reward()
        self.update_latent_phase()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def update_fld_observation_buf(self):
        full_state = torch.cat(
            (
                self.root_states[:, :3] - self.env_origins[:, :3],
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
                self.projected_gravity,
                self.dof_pos,
                self.dof_vel
                ), dim=1
            )
        for key, value in self.decoded_obs_state_idx_dict.items():
            self.fld_state[:, value] = full_state[:, self.fld_state_idx_dict[key]].clone()
        self.fld_observation_buf[:, :-1] = self.fld_observation_buf[:, 1:].clone()
        self.fld_observation_buf[:, -1] = self.fld_state.clone()

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew[name] = rew / self.dt
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        # self.obs_buf = torch.cat((self.obs_buf, self.latent_manifold), dim=-1)
        self.obs_buf = torch.cat((self.obs_buf, torch.sin(2 * torch.pi * self.latent_encoding[:, :, 0])), dim=-1)
        self.obs_buf = torch.cat((self.obs_buf, torch.cos(2 * torch.pi * self.latent_encoding[:, :, 0])), dim=-1)
        self.obs_buf = torch.cat((self.obs_buf, (self.latent_encoding[:, :, 1:].swapaxes(1, 2).flatten(1, 2) - self.latent_param_mean) / self.latent_param_std), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def update_task_sampler_curriculum(self, env_ids):
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        self.task_sampler_curriculum_flag = True
        for name in self.tracking_reconstructed_terms:
            self.task_sampler_curriculum_flag &= (torch.mean(self.episode_sums[name][env_ids]) / self.max_episode_length
            > self.cfg.task_sampler.curriculum_performance_threshold * self.reward_scales[name])
        
    def get_fld_latent_param_statistics(self):
        return self.latent_param_max, self.latent_param_min, self.latent_param_mean, self.latent_param_std

    def get_library_size(self):
        return len(self.task_library)

    def get_libraries(self):
        return self.task_library, self.performance_library

    def empty_libraries(self):
        self.task_library = torch.empty(0, self.cfg.fld.latent_channel * 3, device=self.device, dtype=torch.float, requires_grad=False)
        self.performance_library = torch.empty(0, device=self.device, dtype=torch.float, requires_grad=False)

    def get_elite_buffer_size(self):
        return len(self.elite_task_buffer)

    def get_elite_buffers(self):
        return self.elite_task_buffer, self.elite_performance_buffer

    def empty_elite_buffers(self):
        self.elite_task_buffer = torch.empty(0, self.cfg.fld.latent_channel * 3, device=self.device, dtype=torch.float, requires_grad=False)
        self.elite_performance_buffer = torch.empty(0, device=self.device, dtype=torch.float, requires_grad=False)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12:30] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[30:48] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[48:66] = 0.0  # previous actions
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.shoulder_yaw_indices = torch.tensor([i for i in range(self.num_dof) if "shoulder_yaw" in self.dof_names[i]], device=self.device, requires_grad=False)
        self.shoulder_abad_indices = torch.tensor([i for i in range(self.num_dof) if "shoulder_abad" in self.dof_names[i]], device=self.device, requires_grad=False)
        self.shoulder_pitch_indices = torch.tensor([i for i in range(self.num_dof) if "shoulder_pitch" in self.dof_names[i]], device=self.device, requires_grad=False)
        self.elbow_indices = torch.tensor([i for i in range(self.num_dof) if "elbow" in self.dof_names[i]], device=self.device, requires_grad=False)
        self.arm_joint_indices = torch.cat(
            (
                self.shoulder_yaw_indices,
                self.shoulder_abad_indices,
                self.shoulder_pitch_indices,
                self.elbow_indices,
                )
            )
        self.hip_yaw_indices = torch.tensor([i for i in range(self.num_dof) if "hip_yaw" in self.dof_names[i]], device=self.device, requires_grad=False)
        self.hip_abad_indices = torch.tensor([i for i in range(self.num_dof) if "hip_abad" in self.dof_names[i]], device=self.device, requires_grad=False)
        self.hip_pitch_indices = torch.tensor([i for i in range(self.num_dof) if "hip_pitch" in self.dof_names[i]], device=self.device, requires_grad=False)
        self.knee_indices = torch.tensor([i for i in range(self.num_dof) if "knee" in self.dof_names[i]], device=self.device, requires_grad=False)
        self.ankle_indices = torch.tensor([i for i in range(self.num_dof) if "ankle" in self.dof_names[i]], device=self.device, requires_grad=False)
        self.leg_joint_indices = torch.cat(
            (
                self.hip_yaw_indices,
                self.hip_abad_indices,
                self.hip_pitch_indices,
                self.knee_indices,
                self.ankle_indices
                )
            )
        self.latent_encoding = torch.zeros(self.num_envs, self.cfg.fld.latent_channel, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.params = torch.zeros(self.num_envs, self.cfg.fld.latent_channel, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.latent_manifold = torch.zeros(self.num_envs, self.cfg.fld.latent_channel * 2, dtype=torch.float, device=self.device, requires_grad=False)
        # keyboard controller
        self.latent_variable_list = ['phase', 'frequency', 'amplitude', 'offset']
        self.latent_variable_selector = torch.tensor([-1], device=self.device, dtype=torch.long, requires_grad=False)
        self.latent_channel_selector = torch.tensor([-1], device=self.device, dtype=torch.long, requires_grad=False)
        self.latent_value_modifier = torch.tensor([0.0], device=self.device, dtype=torch.float, requires_grad=False)
        self.task_library = torch.empty(0, self.cfg.fld.latent_channel * 3, device=self.device, dtype=torch.float, requires_grad=False)
        self.performance_library = torch.empty(0, device=self.device, dtype=torch.float, requires_grad=False)
        self.elite_task_buffer = torch.empty(0, self.cfg.fld.latent_channel * 3, device=self.device, dtype=torch.float, requires_grad=False)
        self.elite_performance_buffer = torch.empty(0, device=self.device, dtype=torch.float, requires_grad=False)

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        self.rew = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) for name in self.reward_scales.keys()}
        self.tracking_reconstructed_terms = [name for name in self.reward_scales.keys() if "tracking_reconstructed" in name]

    def _sample_latent_encoding(self, env_ids):
        if len(env_ids) == 0:
            return
        self.latent_encoding[env_ids, :, 0] = torch_rand_float(
            -0.5,
            0.5,
            (len(env_ids), self.fld_latent_channel),
            device=self.device,
            )
        self.latent_encoding[env_ids, :, 1:] = self.task_sampler.sample(len(env_ids)).view(len(env_ids), 3, self.fld_latent_channel).swapaxes(1, 2)
    
    def get_latent_encoding_from_transitions(self, full_state_transitions):
        state_transitions = (full_state_transitions[:, :, self.fld_dim_of_interest] - self.state_transitions_mean) / self.state_transitions_std
        with torch.no_grad():
            _, _, _, params = self.fld(state_transitions.swapaxes(1, 2))
        latent_encoding = torch.cat([param.unsqueeze(-1) for param in params], dim=-1)
        return latent_encoding

    def get_latent_dynamics_error(self, full_state_transitions, k):
        state_transitions = (full_state_transitions[:, :, self.fld_dim_of_interest] - self.state_transitions_mean) / self.state_transitions_std
        error = self.fld.get_dynamics_error(state_transitions, k)
        return error
    
    def _get_keyboard_events(self) -> Dict[str, KeyboardAction]:
        """Simple keyboard controller for linear and angular velocity."""

        def print_selector():
            print(f"latent_variable_selector: {self.latent_variable_selector}")
            print(f"latent_channel_selector: {self.latent_channel_selector}")
        
        def print_command():
            latent_variable = self.latent_variable_list[self.latent_variable_selector]
            print(f"{latent_variable}: {self.latent_encoding[0, :, self.latent_variable_selector]}")
            
        key_board_events = {
            "p": Switch("latent_variable", start_state=-1, toggle_state=0, variable_reference=self.latent_variable_selector, callback=print_selector),
            "f": Switch("latent_variable", start_state=-1, toggle_state=1, variable_reference=self.latent_variable_selector, callback=print_selector),
            "a": Switch("latent_variable", start_state=-1, toggle_state=2, variable_reference=self.latent_variable_selector, callback=print_selector),
            "o": Switch("latent_variable", start_state=-1, toggle_state=3, variable_reference=self.latent_variable_selector, callback=print_selector),
            "u": Delta("latent_value_modifier", amount=0.1, variable_reference=self.latent_value_modifier, callback=print_command),
            "j": Delta("latent_value_modifier", amount=-0.1, variable_reference=self.latent_value_modifier, callback=print_command),
        }
        for i in range(self.cfg.fld.latent_channel):
            key_board_events[f"{i}"] = Switch("latent_channel", start_state=-1, toggle_state=i, variable_reference=self.latent_channel_selector, callback=print_selector)

        return key_board_events

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)
        if self.cfg.task_sampler.curriculum and self.common_step_counter % self.max_episode_length == 0:
            self.update_task_sampler_curriculum(env_ids)
        if self.cfg.task_sampler.collect_samples and self.common_step_counter % self.cfg.task_sampler.collect_sample_step_interval == 0:
            self.collect_task_samples(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._sample_latent_encoding(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def collect_task_samples(self, env_ids):
        env_ids_mask = torch.ones(len(env_ids), device=self.device, dtype=torch.bool, requires_grad=False)
        performance_scale_avg = torch.zeros(len(env_ids), device=self.device, dtype=torch.float, requires_grad=False)
        for name in self.tracking_reconstructed_terms:
            performance_scale = (self.episode_sums[name][env_ids] / self.max_episode_length) / self.reward_scales[name]
            env_ids_mask &= (performance_scale > self.cfg.task_sampler.collect_elite_performance_threshold)
            performance_scale_avg += performance_scale
        performance_scale_avg /= len(self.tracking_reconstructed_terms)
        self.update_task_libraries(env_ids, performance_scale_avg, env_ids_mask)

    def update_task_libraries(self, env_ids, performance, env_ids_mask):
        self.task_library = torch.cat((self.task_library, self.latent_encoding[env_ids, :, 1:].swapaxes(1, 2).flatten(1, 2)))[-self.cfg.task_sampler.library_size:]
        self.performance_library = torch.cat((self.performance_library, performance))[-self.cfg.task_sampler.library_size:]
        elite_env_ids = env_ids[env_ids_mask]
        self.elite_task_buffer = torch.cat((self.elite_task_buffer, self.latent_encoding[elite_env_ids, :, 1:].swapaxes(1, 2).flatten(1, 2)))[-self.cfg.task_sampler.elite_buffer_size:]
        self.elite_performance_buffer = torch.cat((self.elite_performance_buffer, performance[env_ids_mask]))[-self.cfg.task_sampler.elite_buffer_size:]

    def update_latent_phase(self):
        self.latent_encoding[:, :, 0] += self.latent_encoding[:, :, 1] * self.dt
        self.latent_encoding[:, :, 0] = (self.latent_encoding[:, :, 0] + 0.5) % 1.0 - 0.5
        noise_level = self.cfg.domain_rand.latent_encoding_update_noise_level
        latent_param = self.latent_encoding[:, :, 1:].swapaxes(1, 2).flatten(1, 2)
        latent_param += torch.randn_like(latent_param, device=self.device, dtype=torch.float, requires_grad=False) * self.latent_param_std * noise_level
        self.latent_encoding[:, :, 1:] = latent_param.view(self.num_envs, 3, self.fld_latent_channel).swapaxes(1, 2)
        
        phase = self.latent_encoding[:, :, 0]
        frequency = self.latent_encoding[:, :, 1]
        amplitude = self.latent_encoding[:, :, 2]
        offset = self.latent_encoding[:, :, 3]
        reconstructed_z = amplitude.unsqueeze(-1) * torch.sin(2 * torch.pi * (frequency.unsqueeze(-1) * self.fld.args + phase.unsqueeze(-1))) + offset.unsqueeze(-1)
        with torch.no_grad():
            decoded_obs_buf_pred = self.fld.decoder(reconstructed_z)
        decoded_obs_buf_raw = decoded_obs_buf_pred.swapaxes(1, 2)
        self.decoded_obs[:] = decoded_obs_buf_raw[:, -1, :] * self.state_transitions_std + self.state_transitions_mean
        self.latent_manifold[:] = torch.hstack(
            (
                amplitude * torch.sin(2.0 * torch.pi * phase) + offset,
                amplitude * torch.cos(2.0 * torch.pi * phase) + offset,
                )
            )

    def get_latent_params(self):
        fld_observation_buf_standardized = (self.fld_observation_buf - self.state_transitions_mean) / self.state_transitions_std
        with torch.no_grad():
            _, _, _, params = self.fld(fld_observation_buf_standardized.swapaxes(1, 2))
        self.params[:] = torch.cat([param.unsqueeze(-1) for param in params], dim=-1)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _reward_arm_near_home(self):
        return torch.sum(torch.abs(self.dof_pos[:, self.arm_joint_indices] - self.default_dof_pos[:, self.arm_joint_indices]), dim=-1)

    def _reward_leg_near_home(self):
        return torch.sum(torch.abs(self.dof_pos[:, self.leg_joint_indices] - self.default_dof_pos[:, self.leg_joint_indices]), dim=-1)

    def _reward_tracking_reconstructed_lin_vel(self):
        error = torch.sum(torch.square((self.decoded_obs - self.fld_state)[:, torch.tensor(self.decoded_obs_state_idx_dict["base_lin_vel"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
        return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_lin_vel_scale)

    def _reward_tracking_reconstructed_ang_vel(self):
        error = torch.sum(torch.square((self.decoded_obs - self.fld_state)[:, torch.tensor(self.decoded_obs_state_idx_dict["base_ang_vel"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
        return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_ang_vel_scale)

    def _reward_tracking_reconstructed_projected_gravity(self):
        error = torch.sum(torch.square((self.decoded_obs - self.fld_state)[:, torch.tensor(self.decoded_obs_state_idx_dict["projected_gravity"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
        return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_projected_gravity_scale)

    def _reward_tracking_reconstructed_dof_pos_leg_l(self):
        error = torch.sum(torch.square((self.decoded_obs - self.fld_state)[:, torch.tensor(self.decoded_obs_state_idx_dict["dof_pos_leg_l"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
        return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_leg_l_scale)

    def _reward_tracking_reconstructed_dof_pos_arm_l(self):
        error = torch.sum(torch.square((self.decoded_obs - self.fld_state)[:, torch.tensor(self.decoded_obs_state_idx_dict["dof_pos_arm_l"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
        return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_arm_l_scale)

    def _reward_tracking_reconstructed_dof_pos_leg_r(self):
        error = torch.sum(torch.square((self.decoded_obs - self.fld_state)[:, torch.tensor(self.decoded_obs_state_idx_dict["dof_pos_leg_r"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
        return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_leg_r_scale)

    def _reward_tracking_reconstructed_dof_pos_arm_r(self):
        error = torch.sum(torch.square((self.decoded_obs - self.fld_state)[:, torch.tensor(self.decoded_obs_state_idx_dict["dof_pos_arm_r"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
        return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_arm_r_scale)
