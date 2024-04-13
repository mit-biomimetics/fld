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
MOVE_CAMERA = True


from learning.modules.fld import FLD
from learning.modules.plotter import Plotter
import matplotlib.pyplot as plt


def replay(args: argparse.Namespace):

    args.task = "mit_humanoid"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
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
    
    env_cfg.fld.load_root = None  

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

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

    datasets_root = os.path.join(LEGGED_GYM_ROOT_DIR + "/resources/robots/mit_humanoid/datasets/misc")
    motion_data = os.listdir(datasets_root)
    motion_name_set = [data.replace('motion_data_', '').replace('.pt', '') for data in motion_data if "combined" not in data and ".pt" in data]
    aggregated_dataset_collection = []

    for i, motion_name in enumerate(motion_name_set):
        motion_path = os.path.join(datasets_root, "motion_data_" + motion_name + ".pt")
        motion_data = torch.load(motion_path, map_location=env.device)[:, :, :]   
        loaded_num_trajs, loaded_num_steps, loaded_state_dim = motion_data.size()
        print(f"[Motion Loader] Loaded motion {motion_name} with {loaded_num_trajs} trajectories, {loaded_num_steps} steps with {loaded_state_dim} dimensions.")

        aggregated_dataset_collection.append(motion_data.unsqueeze(0))

    aggregated_dataset_collection = torch.cat(aggregated_dataset_collection, dim=0)   

    camera_rot = 0
    camera_rot_per_sec = np.pi / 6
    
    for dataset_idx in range(aggregated_dataset_collection.shape[0]):
        for traj_idx in range(aggregated_dataset_collection.shape[1]):
            for step_idx in range(aggregated_dataset_collection.shape[2]):

                # print(f"Playing dataset {motion_name_set[dataset_idx]}, trajectory {traj_idx}.")

                env.render()
                env.gym.simulate(env.sim)

                root_pos = aggregated_dataset_collection[dataset_idx, traj_idx, step_idx, 0:3].unsqueeze(0)
                root_ori = aggregated_dataset_collection[dataset_idx, traj_idx, step_idx, 3:7].unsqueeze(0)
                root_lin_vel = aggregated_dataset_collection[dataset_idx, traj_idx, step_idx, 7:10].unsqueeze(0)
                root_ang_vel = aggregated_dataset_collection[dataset_idx, traj_idx, step_idx, 10:13].unsqueeze(0)
                root_lin_vel = quat_rotate(root_ori, root_lin_vel) 
                root_ang_vel = quat_rotate(root_ori, root_ang_vel)
                dof_pos = aggregated_dataset_collection[dataset_idx, traj_idx, step_idx, 16:34].unsqueeze(0)
                dof_vel = aggregated_dataset_collection[dataset_idx, traj_idx, step_idx, 34:52].unsqueeze(0)

                step_idx += 1

                root_pos[:, :2] = root_pos[:, :2] + env.env_origins[:, :2]  # add xy offset
                env.root_states[:, :3] = root_pos   
                env.root_states[:, 3:7] = root_ori
                env.root_states[:, 7:10] = root_lin_vel
                env.root_states[:, 10:13] = root_ang_vel

                env.dof_state[:, 0] = dof_pos.squeeze(0)
                env.dof_state[:, 1] = dof_vel.squeeze(0)

                env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32) 
                env.gym.set_dof_state_tensor_indexed(env.sim,   
                                                    gymtorch.unwrap_tensor(env.dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
                env.gym.set_actor_root_state_tensor_indexed(env.sim,    
                                                            gymtorch.unwrap_tensor(env.root_states),
                                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
                
                if MOVE_CAMERA:
                    look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
                    camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
                    camera_relative_position = 2.0 * np.array([np.cos(camera_rot), np.sin(camera_rot), 0.45])
                    env.set_camera(look_at + camera_relative_position, look_at)


if __name__ == "__main__":
    args = get_args()
    replay(args)
