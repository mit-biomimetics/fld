"""
Plays a trained policy and logs statistics.
"""

# humanoid-gym
from humanoid_gym import LEGGED_GYM_ROOT_DIR
from humanoid_gym.envs import task_registry
from humanoid_gym.utils import get_args, export_policy_as_jit, export_policy_as_onnx, Logger
from humanoid_gym.utils.keyboard_controller import KeyBoardController, KeyboardAction, Delta, Switch

# isaac-gym
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    quat_rotate,
)

# learning
from learning.datasets.motion_loader import MotionLoader

# python
import argparse
import os
import numpy as np
import torch

# global settings
EXPORT_POLICY = True
RECORD_FRAMES = False
MOVE_CAMERA = False
PLAY_LOADED_DATA = True
PLOT = True

from learning.modules.fld import FLD
from learning.modules.plotter import Plotter
import matplotlib.pyplot as plt


def preview(args: argparse.Namespace):
    args.task = "mit_humanoid"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 2)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    env_cfg.commands.resampling_time = 1000
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0.0, 0.0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    motion_file = LEGGED_GYM_ROOT_DIR + "/resources/robots/mit_humanoid/datasets/decoded/motion_data.pt"
    motion_loader = MotionLoader(
    device=env.device,
    motion_file=motion_file,
    reference_observation_horizon=env.fld_observation_horizon,
    )
    def _zero_torques(self, actions):
        return torch.zeros_like(actions)
    
    env._compute_torques = type(env._compute_torques)(_zero_torques, env)

    state_idx_dict = {
        "base_pos": [0, 1, 2],
        "base_quat": [3, 4, 5, 6],
        "base_lin_vel": [7, 8, 9],
        "base_ang_vel": [10, 11, 12],
        "projected_gravity": [13, 14, 15],
        "dof_pos_leg_l": [16, 17, 18, 19, 20],
        "dof_pos_arm_l": [21, 22, 23, 24],
        "dof_pos_leg_r": [25, 26, 27, 28, 29],
        "dof_pos_arm_r": [30, 31, 32, 33],
    }

    dim_of_interest = torch.cat([torch.tensor(ids, device=env.device, dtype=torch.long, requires_grad=False) for state, ids in state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
    observation_dim = dim_of_interest.size(0)
    observation_horizon = 51
    log_dir_root = LEGGED_GYM_ROOT_DIR + "/logs/flat_mit_humanoid/fld/"
    latent_dim = 8

    fld = FLD(observation_dim, observation_horizon, latent_dim, env.device, encoder_shape=env_cfg.fld.encoder_shape, decoder_shape=env_cfg.fld.decoder_shape)

    runs = os.listdir(log_dir_root)
    runs.sort()
    last_run = os.path.join(log_dir_root, runs[-1])
    load_run = last_run
    models = [file for file in os.listdir(load_run) if "model" in file]
    models.sort(key=lambda m: "{0:0>15}".format(m))
    model = models[-1]

    loaded_dict = torch.load(os.path.join(load_run, model))
    fld.load_state_dict(loaded_dict["fld_state_dict"])

    datasets_root = os.path.join(LEGGED_GYM_ROOT_DIR + "/resources/robots/mit_humanoid/datasets/misc")
    motion_data = os.listdir(datasets_root)
    motion_name_set = [data.replace('motion_data_', '').replace('.pt', '') for data in motion_data if "combined" not in data and ".pt" in data]
    aggregated_data_collection = []

    if PLOT:
        plotter = Plotter()
        plt.ion()
        fig1, ax1 = plt.subplots(4, 1)

    for i, motion_name in enumerate(motion_name_set):
        motion_path = os.path.join(datasets_root, "motion_data_" + motion_name + ".pt")
        motion_data = torch.load(motion_path, map_location=env.device)[:, :, dim_of_interest]
        loaded_num_trajs, loaded_num_steps, loaded_obs_dim = motion_data.size()
        print(f"[Motion Loader] Loaded motion {motion_name} with {loaded_num_trajs} trajectories, {loaded_num_steps} steps with {loaded_obs_dim} dimensions.")
        aggregated_data = torch.zeros(loaded_num_trajs,
                                    loaded_num_steps - observation_horizon + 1,
                                    observation_horizon,
                                    loaded_obs_dim,
                                    dtype=torch.float,
                                    device=env.device,
                                    requires_grad=False
                                    )
        for step in range(loaded_num_steps - observation_horizon + 1):
            aggregated_data[:, step] = motion_data[:, step:step+observation_horizon, :]
        aggregated_data_collection.append(aggregated_data.unsqueeze(0))

    aggregated_data_collection = torch.cat(aggregated_data_collection, dim=0)

    state_transitions_mean = aggregated_data_collection.flatten(0, 3).mean(dim=0)
    state_transitions_std = aggregated_data_collection.flatten(0, 3).std(dim=0)
    state_transitions_data = (aggregated_data_collection - state_transitions_mean) / state_transitions_std

    fld.eval()

    eval_traj = state_transitions_data[0, 0].swapaxes(1, 2)
    with torch.no_grad():
        _, _, _, params = fld(eval_traj)
        env.latent_sample_phase = params[0][0, :]
        env.latent_sample_frequency = params[1][0, :]
        env.latent_sample_amplitude = params[2][0, :]
        env.latent_sample_offset = params[3][0, :]
    
    
    latent_variable_list = ['phase', 'frequency', 'amplitude', 'offset']
    env.latent_variable_selector = torch.tensor([-1], device=env.device, dtype=torch.long, requires_grad=False)
    env.latent_channel_selector = torch.tensor([-1], device=env.device, dtype=torch.long, requires_grad=False)
    env.latent_value_modifier = torch.tensor([0.0], device=env.device, dtype=torch.float, requires_grad=False)
    
    def print_selector():
        print(f"latent_variable_selector: {env.latent_variable_selector}")
        print(f"latent_channel_selector: {env.latent_channel_selector}")
    
    def print_command():
        latent_variable = latent_variable_list[env.latent_variable_selector]
        latent_variable_name = "latent_sample_" + str(latent_variable)
        print(f"{latent_variable}: {getattr(env, latent_variable_name)}")
        
    key_board_events = {
        "p": Switch("latent_variable", start_state=0, toggle_state=0, variable_reference=env.latent_variable_selector, callback=print_selector),
        "f": Switch("latent_variable", start_state=1, toggle_state=1, variable_reference=env.latent_variable_selector, callback=print_selector),
        "a": Switch("latent_variable", start_state=2, toggle_state=2, variable_reference=env.latent_variable_selector, callback=print_selector),
        "o": Switch("latent_variable", start_state=3, toggle_state=3, variable_reference=env.latent_variable_selector, callback=print_selector),
        "u": Delta("latent_value_modifier", amount=0.1, variable_reference=env.latent_value_modifier, callback=print_command),
        "j": Delta("latent_value_modifier", amount=-0.1, variable_reference=env.latent_value_modifier, callback=print_command),
    }
    for i in range(latent_dim):
        key_board_events[f"{i}"] = Switch("latent_channel", start_state=i, toggle_state=i, variable_reference=env.latent_channel_selector, callback=print_selector)

    env.keyboard_controller = KeyBoardController(env, key_board_events)
    env.keyboard_controller.print_options()

    
    for i in range(10 * int(env.max_episode_length)):
        env.update_keyboard_events()
        getattr(env, f"latent_sample_{latent_variable_list[env.latent_variable_selector]}")[env.latent_channel_selector] += env.latent_value_modifier
        env.latent_value_modifier[:] = 0.0
        # actions = policy(obs.detach())
        actions = torch.zeros_like(env.actions)
        # obs, _, rews, dones, infos = env.step(actions.detach())
        env.render()
        env.gym.simulate(env.sim)

        env.latent_sample_phase += env.latent_sample_frequency * env.dt
        latent_sample_z = env.latent_sample_amplitude.unsqueeze(-1) * torch.sin(2 * torch.pi * (env.latent_sample_frequency.unsqueeze(-1) * fld.args + env.latent_sample_phase.unsqueeze(-1))) + env.latent_sample_offset.unsqueeze(-1)
        with torch.no_grad():
            decoded_traj_pred = fld.decoder(latent_sample_z.unsqueeze(0))
        decoded_traj_raw = decoded_traj_pred.swapaxes(1, 2)
        decoded_traj = decoded_traj_raw * state_transitions_std + state_transitions_mean
        
        decoded_traj = decoded_traj[0, 0, :]
        
        decoded_traj_buf = torch.zeros(52, device=env.device, dtype=torch.float, requires_grad=False)
        decoded_traj_buf[2] = 0.66
        decoded_traj_buf[6] = 1.0

        decoded_traj_buf[dim_of_interest] = decoded_traj

        if PLOT:
            plotter.plot_circles(ax1[0], env.latent_sample_phase, env.latent_sample_amplitude, title="Learned Phase Timing"  + " " + str(latent_dim) + "x" + str(2), show_axes=False)
            plotter.plot_curves(ax1[1], latent_sample_z, -1.0, 1.0, -2.0, 2.0, title="Latent Parametrized Signal" + " " + str(latent_dim) + "x" + str(observation_horizon), show_axes=False)
            plotter.plot_curves(ax1[2], decoded_traj_pred.squeeze(0), -1.0, 1.0, -5.0, 5.0, title="Curve Reconstruction" + " " + str(fld.input_channel) + "x" + str(observation_horizon), show_axes=False)
            plotter.plot_curves(ax1[3], decoded_traj_pred.flatten(1, 2), -1.0, 1.0, -5.0, 5.0, title="Curve Reconstruction (Flattened)" + " " + str(1) + "x" + str(fld.input_channel*observation_horizon), show_axes=False)
            fig1.canvas.draw()
            fig1.canvas.flush_events()

        if PLAY_LOADED_DATA:
            frames = decoded_traj_buf.repeat(env.num_envs, 1)
            env.dof_pos[:] = motion_loader.get_dof_pos(frames)
            env.dof_vel[:] = motion_loader.get_dof_vel(frames)
            root_pos = motion_loader.get_base_pos(frames)
            root_pos[:, :2] = root_pos[:, :2] + env.env_origins[:, :2]
            env.root_states[:, :3] = root_pos
            root_ori = motion_loader.get_base_quat(frames)
            env.root_states[:, 3:7] = root_ori
            env.root_states[:, 7:10] = quat_rotate(root_ori, motion_loader.get_base_lin_vel(frames))
            env.root_states[:, 10:13] = quat_rotate(root_ori, motion_loader.get_base_ang_vel(frames))

            env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32)
            env.gym.set_dof_state_tensor_indexed(env.sim,
                                                gymtorch.unwrap_tensor(env.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

            env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                        gymtorch.unwrap_tensor(env.root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


if __name__ == "__main__":
    args = get_args()
    preview(args)
