from humanoid_gym import LEGGED_GYM_ROOT_DIR
from isaacgym.torch_utils import (
    quat_mul,
    quat_conjugate,
    normalize,
    quat_from_angle_axis,
)
import os
import json
import torch

class MotionLoader:

    def __init__(self, device, motion_file=None, corruption_level=0.0, reference_observation_horizon=2, test_mode=False, test_observation_dim=None):
        self.device = device
        self.reference_observation_horizon = reference_observation_horizon
        if motion_file is None:
            motion_file = LEGGED_GYM_ROOT_DIR + "/resources/robots/anymal_c/datasets/motion_data.pt"
        self.reference_state_idx_dict_file = os.path.join(os.path.dirname(motion_file), "reference_state_idx_dict.json")
        with open(self.reference_state_idx_dict_file, 'r') as f:
            self.state_idx_dict = json.load(f)
        self.observation_dim = sum([ids[1] - ids[0] for state, ids in self.state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
        self.observation_start_dim = self.state_idx_dict["base_lin_vel"][0]
        loaded_data = torch.load(motion_file, map_location=self.device)

        # Normalize and standardize quaternions
        base_quat = normalize(loaded_data[:, :, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]])
        base_quat[base_quat[:, :, -1] < 0] = -base_quat[base_quat[:, :, -1] < 0]
        loaded_data[:, :, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]] = base_quat

        # Load data for DTW
        motion_file_dtw = os.path.join(os.path.dirname(motion_file), "motion_data_original.pt")
        try:
            self.dtw_reference = torch.load(motion_file_dtw, map_location=self.device)[:, :, self.observation_start_dim:]
            print(f"[MotionLoader] Loaded DTW reference motion clips.")
        except:
            self.dtw_reference = None
            print(f"[MotionLoader] No DTW reference motion clips provided.")

        self.data = self._data_corruption(loaded_data, level=corruption_level)
        self.num_motion_clips, self.num_steps, self.reference_full_dim = self.data.size()
        print(f"[MotionLoader] Loaded {self.num_motion_clips} motion clips from {motion_file}. Each records {self.num_steps} steps and {self.reference_full_dim} states.")

        # Preload transitions
        self.num_preload_transitions = 500000
        motion_clip_sample_ids = torch.randint(0, self.num_motion_clips, (self.num_preload_transitions,), device=self.device)
        step_sample = torch.rand(self.num_preload_transitions, device=self.device) * (self.num_steps - self.reference_observation_horizon)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.reference_full_dim,
            dtype=torch.float,
            device=self.device,
            requires_grad=False
        )
        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self._get_frame_at_step(motion_clip_sample_ids, step_sample + i)
        
        if test_mode:
            self.observation_dim = test_observation_dim

    def _data_corruption(self, loaded_data, level=0):
        if level == 0:
            print(f"[MotionLoader] Proceeded without processing the loaded data.")
        else:
            loaded_data = self._rand_dropout(loaded_data, level)
            loaded_data = self._rand_noise(loaded_data, level)
            loaded_data = self._rand_interpolation(loaded_data, level)
            loaded_data = self._rand_duplication(loaded_data, level)
        return loaded_data

    def _rand_dropout(self, data, level=0):
        num_motion_clips, num_steps, reference_full_dim = data.size()
        num_dropouts = round(num_steps * level)
        if num_dropouts == 0:
            return data
        dropped_data = torch.zeros(num_motion_clips, num_steps - num_dropouts, reference_full_dim, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(num_motion_clips):
            step_ids = torch.randperm(num_steps)[:-num_dropouts].sort()[0]
            dropped_data[i] = data[i, step_ids]
        return dropped_data

    def _rand_interpolation(self, data, level=0):
        num_motion_clips, num_steps, reference_full_dim = data.size()
        num_interpolations = round((num_steps - 2) * level)
        if num_interpolations == 0:
            return data
        interpolated_data = data
        for i in range(num_motion_clips):
            step_ids = torch.randperm(num_steps)
            step_ids = step_ids[(step_ids != 0) * (step_ids != num_steps - 1)]
            step_ids = step_ids[:num_interpolations].sort()[0]
            interpolated_data[i, step_ids] = self.slerp(data[i, step_ids - 1], data[i, step_ids + 1], 0.5)
            interpolated_data[i, step_ids, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]] = self.quaternion_slerp(
                data[i, step_ids - 1, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]], 
                data[i, step_ids + 1, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]], 
                0.5
            )
        return interpolated_data

    def _rand_duplication(self, data, level=0):
        num_motion_clips, num_steps, reference_full_dim = data.size()
        num_duplications = round(num_steps * level) * 10
        if num_duplications == 0:
            return data
        duplicated_data = torch.zeros(num_motion_clips, num_steps + num_duplications, reference_full_dim, dtype=torch.float, device=self.device, requires_grad=False)
        step_ids = torch.randint(0, num_steps, (num_motion_clips, num_duplications), device=self.device)
        for i in range(num_motion_clips):
            duplicated_step_ids = torch.cat((torch.arange(num_steps, device=self.device), step_ids[i])).sort()[0]
            duplicated_data[i] = data[i, duplicated_step_ids]
        return duplicated_data

    def _rand_noise(self, data, level=0):
        noise_scales_dict = {
            "base_pos": 0.1,
            "base_quat": 0.01,
            "base_lin_vel": 0.1,
            "base_ang_vel": 0.2,
            "projected_gravity": 0.05,
            "base_height": 0.1,
            "dof_pos": 0.01,
            "dof_vel": 1.5
        }
        noise_scale_vec = torch.zeros_like(data[0, 0], device=self.device, dtype=torch.float, requires_grad=False)
        for key, value in self.state_idx_dict.items():
            noise_scale_vec[value[0]:value[1]] = noise_scales_dict[key] * level
        data += (2 * torch.randn_like(data) - 1) * noise_scale_vec
        return data

    def _get_frame_at_step(self, motion_clip_sample_ids, step_sample):
        step_low, step_high = step_sample.floor().long(), step_sample.ceil().long()
        blend = (step_sample - step_low).unsqueeze(-1)
        frame = self.slerp(self.data[motion_clip_sample_ids, step_low], self.data[motion_clip_sample_ids, step_high], blend)
        frame[:, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]] = self.quaternion_slerp(
            self.data[motion_clip_sample_ids, step_low, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]], 
            self.data[motion_clip_sample_ids, step_high, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]], 
            blend
            )
        return frame

    def get_frames(self, num_frames):
        ids = torch.randint(0, self.num_preload_transitions, (num_frames,), device=self.device)
        return self.preloaded_states[ids, 0]
    
    def get_transitions(self, num_transitions):
        ids = torch.randint(0, self.num_preload_transitions, (num_transitions,), device=self.device)
        return self.preloaded_states[ids, :]
    
    def slerp(self, value_low, value_high, blend):
        return (1.0 - blend) * value_low + blend * value_high

    def quaternion_slerp(self, quat_low, quat_high, blend):
        relative_quat = normalize(quat_mul(quat_high, quat_conjugate(quat_low)))
        angle = 2 * torch.acos(relative_quat[:, -1]).unsqueeze(-1)
        axis = normalize(relative_quat[:, :3])
        angle_slerp = self.slerp(torch.zeros_like(angle), angle, blend).squeeze(-1)
        relative_quat_slerp = quat_from_angle_axis(angle_slerp, axis)        
        return normalize(quat_mul(relative_quat_slerp, quat_low))

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim:]
            yield states

    def get_base_pos(self, frames):
        if "base_pos" in self.state_idx_dict:
            return frames[:, self.state_idx_dict["base_pos"][0]:self.state_idx_dict["base_pos"][1]]
        else:
            raise Exception("[MotionLoader] base_pos not specified in the state_idx_dict")

    def get_base_quat(self, frames):
        if "base_quat" in self.state_idx_dict:
            return frames[:, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]]
        else:
            raise Exception("[MotionLoader] base_quat not specified in the state_idx_dict")

    def get_base_lin_vel(self, frames):
        if "base_lin_vel" in self.state_idx_dict:
            return frames[:, self.state_idx_dict["base_lin_vel"][0]:self.state_idx_dict["base_lin_vel"][1]]
        else:
            raise Exception("[MotionLoader] base_lin_vel not specified in the state_idx_dict")

    def get_base_ang_vel(self, frames):
        if "base_ang_vel" in self.state_idx_dict:
            return frames[:, self.state_idx_dict["base_ang_vel"][0]:self.state_idx_dict["base_ang_vel"][1]]
        else:
            raise Exception("[MotionLoader] base_ang_vel not specified in the state_idx_dict")

    def get_projected_gravity(self, frames):
        if "projected_gravity" in self.state_idx_dict:
            return frames[:, self.state_idx_dict["projected_gravity"][0]:self.state_idx_dict["projected_gravity"][1]]
        else:
            raise Exception("[MotionLoader] projected_gravity not specified in the state_idx_dict")

    def get_dof_pos(self, frames):
        if "dof_pos" in self.state_idx_dict:
            return frames[:, self.state_idx_dict["dof_pos"][0]:self.state_idx_dict["dof_pos"][1]]
        else:
            raise Exception("[MotionLoader] dof_pos not specified in the state_idx_dict")

    def get_dof_vel(self, frames):
        if "dof_vel" in self.state_idx_dict:
            return frames[:, self.state_idx_dict["dof_vel"][0]:self.state_idx_dict["dof_vel"][1]]
        else:
            raise Exception("[MotionLoader] dof_vel not specified in the state_idx_dict")

    def get_feet_pos(self, frames):
        if "feet_pos" in self.state_idx_dict:
            return frames[:, self.state_idx_dict["feet_pos"][0]:self.state_idx_dict["feet_pos"][1]]
        else:
            raise Exception("[MotionLoader] feet_pos not specified in the state_idx_dict")
