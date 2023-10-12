#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# isaacgym
from isaacgym import gymapi
from isaacgym import gymutil

# python
import os
import copy
import torch
import numpy as np
import random
import argparse


"""
Dictionary <-> Class operations.
"""


def class_to_dict(obj) -> dict:
    """Convert a class to dictionary data.

    Args:
        obj ([type]): Input class instance.

    Returns:
        dict: Dictionary containing values from the class.
    """
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, d: dict) -> None:
    """Updates a class with values from dictionary.

    Args:
        obj ([type]): Input class instance.
        d (dict): Dictionary to update values from.
    """
    for key, val in d.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


"""
Parsing configurations.
"""


def parse_sim_params(args: argparse.Namespace, cfg: dict) -> gymapi.SimParams:
    """Parses CLI args and dictionary to generate physics stepping settings.

    Args:
        args (argparse.Namespace): [description]
        cfg (dict): [description]

    Returns:
        gymapi.SimParams: IsaacGym SimParams object with updated settings.
    """
    # Note: Some of the code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def update_cfg_from_args(env_cfg, cfg_train, args: argparse.Namespace) -> tuple:
    """Updates environment and training configrations from CLI args.

    Args:
        env_cfg ([type]): Environment configuration instance.
        cfg_train ([type]): Training configuration instance.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        tuple: Tuple containing the environment and training configurations.
    """
    # environment configuration
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    # training configuration
    if cfg_train is not None:
        # seed
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args() -> argparse.Namespace:
    """Defines custom command-line arguments and parses them.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "anymal_c_flat",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
    ]
    # parse arguments
    args = gymutil.parse_arguments(description="RL Policy using IsaacGym", custom_parameters=custom_parameters)
    # name alignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


"""
MDP-related operations.
"""


def set_seed(seed: int):
    """Set the seeding of the experiment.

    Note:
        If input is -1, then a random integer between (0, 10000) is sampled.

    Args:
        seed (int): The seed value to set.
    """
    # default argument
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("[Utils] Setting seed: {}".format(seed))
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


"""
Model loading and saving.
"""


def get_load_path(root: str, load_run=-1, checkpoint: int = -1) -> str:
    # check if runs present in directory
    try:
        runs = os.listdir(root)
        # TODO: sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except IndexError:
        raise ValueError(f"No runs present in directory: {root}")
    # path to the directory containing the run
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)
    # name of model checkpoint
    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    return os.path.join(load_run, model)


def export_policy_as_jit(actor_critic, path, filename="policy.pt"):
    policy_exporter = TorchPolicyExporter(actor_critic)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(actor_critic, path, filename="policy.onnx"):
    policy_exporter = OnnxPolicyExporter(actor_critic)
    policy_exporter.export(path, filename)


class TorchPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer(
                "hidden_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            self.register_buffer(
                "cell_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            self.forward = self.forward_lstm
            self.reset = self.reset_memory

    def forward_lstm(self, x):
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(x)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class OnnxPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm

    def forward_lstm(self, x_in, h_in, c_in):
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(x)

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=True,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=True,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
