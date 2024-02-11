"""
Plays a trained policy and logs statistics.
"""

# humanoid-gym
from humanoid_gym import LEGGED_GYM_ROOT_DIR
from humanoid_gym.envs import task_registry
from humanoid_gym.utils import get_args, export_policy_as_jit, export_policy_as_onnx, Logger

# python
import argparse
import os
import numpy as np
import torch

# global settings
EXPORT_POLICY = True
MOVE_CAMERA = True

def play(args: argparse.Namespace):
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
    
    env_cfg.domain_rand.added_mass_range = [0.0, 0.0]
    env_cfg.commands.resampling_time = 1000.0
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.3]
    env_cfg.env.episode_length_s = 1000.0
    env_cfg.env.env_spacing = 100.0
    env_cfg.domain_rand.latent_encoding_update_noise_level = 0.0

    # camera
    env_cfg.viewer.pos = [0.0, -2.13, 1.22]
    dir = [0.0, 1.0, -0.4]
    env_cfg.viewer.lookat = [a + b for a, b in zip(env_cfg.viewer.pos, dir)]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    motion_data_collection = []
    datasets_root = os.path.join(LEGGED_GYM_ROOT_DIR + "/resources/robots/mit_humanoid/datasets/misc")
    motion_data = os.listdir(datasets_root)
    motion_name_set = [data.replace('motion_data_', '').replace('.pt', '') for data in motion_data if "combined" not in data and ".pt" in data]
    for i, motion_name in enumerate(motion_name_set):
        motion_path = os.path.join(datasets_root, "motion_data_" + motion_name + ".pt")
        motion_data = torch.load(motion_path, map_location=env.device).to(env.device)
        motion_data_collection.append(motion_data.unsqueeze(0))
    motion_data_collection = torch.cat(motion_data_collection, dim=0)
    motion_data_num_motions, motion_data_num_trajs, motion_data_num_steps, motion_data_observation_dim = motion_data_collection.size()
    motion_idx = 0
    traj_idx = 0
    t = 0
    ood_motion = torch.tensor([], device=env.device, dtype=torch.float, requires_grad=False)
    last_input = ""

    # export policy as a jit module and as onnx model (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        name = "policy"
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, filename=f"{name}.pt")
        export_policy_as_onnx(ppo_runner.alg.actor_critic, path, filename=f"{name}.onnx")
        print("Exported policy to: ", path)

    logger = Logger(env.dt)
    robot_index = 1  # which robot is used for logging
    joint_index = 3  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)

    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        user_input = motion_data_collection[motion_idx, traj_idx, t:t+env.fld_observation_horizon, :]
        if t + env.fld_observation_horizon >= motion_data_num_steps:
            motion_idx = torch.randint(0, motion_data_num_motions, (1,)).item()
            traj_idx = torch.randint(0, motion_data_num_trajs, (1,)).item()
            print(f"[Motion] Motion name {motion_name_set[motion_idx]}")
            print(f"[Motion] Trajectory index {traj_idx}")
            t = 0
        else:
            t += 1

        # compute dynamics error
        dynamics_error = env.get_latent_dynamics_error(user_input.unsqueeze(0), k=1)
        # fallback mechanism
        if dynamics_error < 1.0:
            if last_input != "user":
                print("[Input] User")
            latent_encoding = env.latent_encoding[:].clone()
            latent_encoding[:, :, 1:] = env.get_latent_encoding_from_transitions(user_input.repeat(env.num_envs, 1, 1))[:, :, 1:]
            env.latent_encoding[:] = latent_encoding
            last_input = "user"
        else:
            if last_input != "default":
                print("[Input] Default")
            ood_motion = torch.cat((ood_motion, user_input[[-1]]), dim=0)
            last_input = "default"

        # compute tracking error
        tracking_reconstructed_terms = [name for name in env.reward_scales.keys() if "tracking_reconstructed" in name]
        tracking_error = torch.mean(torch.hstack([env.rew[name].unsqueeze(-1) / env.reward_scales[name] * env.dt for name in tracking_reconstructed_terms]), dim=-1)
        print(f"[FLD] Tracking error: {tracking_error}")
        
        if MOVE_CAMERA:
            camera_position = env.root_states[0, :3].cpu().numpy()
            camera_position[1] -= 2.13
            camera_position[2] = 1.22
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": (actions * env.cfg.control.action_scale + env.default_dof_pos)[robot_index, joint_index].item(),
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    "base_lin_vel": env.fld_state[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["base_lin_vel"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "base_lin_vel_ref": env.decoded_obs[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["base_lin_vel"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "base_ang_vel": env.fld_state[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["base_ang_vel"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "base_ang_vel_ref": env.decoded_obs[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["base_ang_vel"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "projected_gravity": env.fld_state[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["projected_gravity"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "projected_gravity_ref": env.decoded_obs[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["projected_gravity"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "dof_pos_leg_l": env.fld_state[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["dof_pos_leg_l"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "dof_pos_leg_l_ref": env.decoded_obs[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["dof_pos_leg_l"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "dof_pos_arm_l": env.fld_state[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["dof_pos_arm_l"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "dof_pos_arm_l_ref": env.decoded_obs[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["dof_pos_arm_l"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "dof_pos_leg_r": env.fld_state[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["dof_pos_leg_r"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "dof_pos_leg_r_ref": env.decoded_obs[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["dof_pos_leg_r"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "dof_pos_arm_r": env.fld_state[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["dof_pos_arm_r"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                    "dof_pos_arm_r_ref": env.decoded_obs[robot_index, torch.tensor(env.decoded_obs_state_idx_dict["dof_pos_arm_r"], device=env.device, dtype=torch.long, requires_grad=False)].tolist(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    args = get_args()
    play(args)
