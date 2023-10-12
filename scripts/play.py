"""
Plays a trained policy and logs statistics.
"""

# solo-gym
from solo_gym import LEGGED_GYM_ROOT_DIR
from solo_gym.envs import task_registry
from solo_gym.utils import get_args, export_policy_as_jit, export_policy_as_onnx, Logger

# python
import argparse
import os
import numpy as np
import torch

# global settings
EXPORT_POLICY = True
MOVE_CAMERA = True

def play(args: argparse.Namespace):
    args.task = "solo8"
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

    env.keyboard_controller.print_options()

    for i in range(10 * int(env.max_episode_length)):
        env.update_keyboard_events()
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        if MOVE_CAMERA:
            camera_position = env.root_states[0, :3].cpu().numpy()
            camera_position[1] -= 2.0
            camera_position[2] = 1.0
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": (actions * env.cfg.control.action_scale + env.default_dof_pos)[robot_index, joint_index].item(),
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": 0.0,
                    "command_yaw": 0.0,
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
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
