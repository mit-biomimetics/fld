from humanoid_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from humanoid_gym import LEGGED_GYM_ROOT_DIR


class MITHumanoidFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 106
        num_actions = 18

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        terrain_proportions = [0.0, 1.0]
        num_rows = 5
        max_init_terrain_level = 4

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.7] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "left_hip_yaw": 0.0,
            "left_hip_abad": 0.0,
            "left_hip_pitch": -0.4,
            "left_knee": 0.9,
            "left_ankle": -0.45,
            "left_shoulder_pitch": 0.0,
            "left_shoulder_abad": 0.0,
            "left_shoulder_yaw": 0.0,
            "left_elbow": 0.0,

            "right_hip_yaw": 0.0,
            "right_hip_abad": 0.0,
            "right_hip_pitch": -0.4,
            "right_knee": 0.9,
            "right_ankle": -0.45,
            "right_shoulder_pitch": 0.0,
            "right_shoulder_abad": 0.0,
            "right_shoulder_yaw": 0.0,
            "right_elbow": 0.0,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {
            "hip_yaw": 30.0,
            "hip_abad": 30.0,
            "hip_pitch": 30.0,
            "knee": 30.0,
            "ankle": 30.0,
            "shoulder_pitch": 40.0,
            "shoulder_abad": 40.0,
            "shoulder_yaw": 40.0,
            "elbow": 40.0,
            }  # [N*m/rad]
        damping = {
            "hip_yaw": 5.0,
            "hip_abad": 5.0,
            "hip_pitch": 5.0,
            "knee": 5.0,
            "ankle": 5.0,
            "shoulder_pitch": 5.0,
            "shoulder_abad": 5.0,
            "shoulder_yaw": 5.0,
            "elbow": 5.0,
            }  # [N*m*s/rad]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/mit_humanoid/urdf/humanoid_R_sf.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["arm"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.85
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.66
        max_contact_force = 350.0
        tracking_reconstructed_lin_vel_scale = 0.2
        tracking_reconstructed_ang_vel_scale = 0.2
        tracking_reconstructed_projected_gravity_scale = 1.0
        tracking_reconstructed_dof_pos_leg_l_scale = 1.0
        tracking_reconstructed_dof_pos_arm_l_scale = 1.0
        tracking_reconstructed_dof_pos_leg_r_scale = 1.0
        tracking_reconstructed_dof_pos_arm_r_scale = 1.0
        class scales(LeggedRobotCfg.rewards.scales):
            orientation = -0.0
            feet_air_time = 0.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            stand_still = -0.0
            base_height = -0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            tracking_reconstructed_lin_vel = 1.0
            tracking_reconstructed_ang_vel = 1.0
            tracking_reconstructed_projected_gravity = 1.0
            tracking_reconstructed_dof_pos_leg_l = 1.0
            tracking_reconstructed_dof_pos_arm_l = 1.0
            tracking_reconstructed_dof_pos_leg_r = 1.0
            tracking_reconstructed_dof_pos_arm_r = 1.0
            arm_near_home = -0.0
            leg_near_home = -0.0
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        resampling_time = 5.0
        heading_command = False
        class ranges:
            lin_vel_x = [-0.0, 0.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]  # min max [rad/s]

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = True
        max_push_vel_xy = 0.5
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        latent_encoding_update_noise_level = 0.0

    class fld:
        latent_channel = 8
        history_horizon = 51
        encoder_shape = [64, 64]
        decoder_shape = [64, 64]
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
        load_root = LEGGED_GYM_ROOT_DIR + "/logs/flat_mit_humanoid/fld/misc"
        load_model = "model_5000.pt"

    
    class task_sampler:
        collect_samples = True
        collect_sample_step_interval = 5
        collect_elite_performance_threshold = 1.0
        library_size = 5000
        update_interval = 50
        elite_buffer_size = 1
        max_num_updates = 1000
        curriculum = True
        curriculum_scale = 1.25
        curriculum_performance_threshold = 0.8
        max_num_curriculum_updates = 5
        check_update_interval = 100
        class offline:
            pass
        class random:
            pass
        class gmm:
            num_components = 8
        class alp_gmm:
            init_num_components = 8
            min_num_components = 2
            max_num_components = 10
            random_type = "uniform" # "uniform" or "gmm"
        class classifier:
            enabled = False
            num_classes = 9

    
class MITHumanoidFlatCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "FLDOnPolicyRunner"
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 128, 128]
        critic_hidden_dims = [128, 128, 128]
        init_noise_std = 1.0
    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'min_forces'
        experiment_name = 'flat_mit_humanoid'
        algorithm_class_name = "PPO"
        policy_class_name = "ActorCritic"
        task_sampler_class_name = "OfflineSampler" # "OfflineSampler", "GMMSampler", "RandomSampler", "ALPGMMSampler"
        load_run = -1
        max_iterations = 3000