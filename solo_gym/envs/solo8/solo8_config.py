from solo_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from solo_gym import LEGGED_GYM_ROOT_DIR


class Solo8FlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 34
        num_actions = 8

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        terrain_proportions = [0.0, 1.0]
        num_rows = 5
        max_init_terrain_level = 4

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.25]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "FL_HFE": 0.8,
            "HL_HFE": 0.8,
            "FR_HFE": 0.8,
            "HR_HFE": 0.8,

            "FL_KFE": -1.4,
            "HL_KFE": -1.4,
            "FR_KFE": -1.4,
            "HR_KFE": -1.4,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'HFE': 5.0, 'KFE': 5.0} # [N*m/rad]
        damping = {'HFE': 0.1, 'KFE': 0.1} # [N*m*s/rad]
        torque_limit = 2.5

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/solo8/urdf/solo8.urdf'
        foot_name = "FOOT"
        terminate_after_contacts_on = ["base", "UPPER"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.85
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.24
        max_contact_force = 350.0
        only_positive_rewards = True
        class scales(LeggedRobotCfg.rewards.scales):
            ang_vel_xy = -0.0
            base_height = -0.0
            lin_vel_z = -0.0
            orientation = -0.0
            torques = -0.000025
            feet_air_time = 0.1
            tracking_ang_vel = 0.0
            feet_contact_forces = -0.1
            ang_vel_x = -0.1
            ang_vel_z = -0.1
            lin_vel_y = -0.1

    class commands(LeggedRobotCfg.commands):
        num_commands = 1
        curriculum = False
        max_curriculum = 5.0
        resampling_time = 5.0
        heading_command = False
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.2, 0.5]

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = True
        max_push_vel_xy = 0.5
        randomize_base_mass = True
        added_mass_range = [-0.2, 2.0]
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85

    class motion_loader:
        reference_motion_file = LEGGED_GYM_ROOT_DIR + "/resources/robots/solo8/datasets/motion_data.pt"
        corruption_level = 0.0
        reference_observation_horizon = 2
        test_mode = False
        test_observation_dim = None # observation_dim of reference motion

class Solo8FlatCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "WASABIOnPolicyRunner"

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 128, 128]
        critic_hidden_dims = [128, 128, 128]
        init_noise_std = 1.0

    class discriminator:
        reward_coef = 0.1
        reward_lerp = 0.9 # wasabi_reward = (1 - reward_lerp) * style_reward + reward_lerp * task_reward
        style_reward_function = "wasserstein_mapping" # log_mapping, quad_mapping, wasserstein_mapping
        shape = [512, 256]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        wasabi_replay_buffer_size = 1000000
        policy_learning_rate = 1e-3
        discriminator_learning_rate = 5e-7
        discriminator_momentum = 0.5
        discriminator_weight_decay = 1e-3
        discriminator_gradient_penalty_coef = 5
        discriminator_loss_function = "WassersteinLoss" # MSELoss, BCEWithLogitsLoss, WassersteinLoss
        discriminator_num_mini_batches = 80

    class runner(LeggedRobotCfgPPO.runner):
        run_name = "wasabi"
        experiment_name = "flat_solo8"
        algorithm_class_name = "WASABI"
        policy_class_name = "ActorCritic"
        load_run = -1
        max_iterations = 1000
        normalize_style_reward = True
        compute_dynamic_time_warping = False
        num_dynamic_time_warping_samples = 50
