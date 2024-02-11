#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# python
import argparse
import os
from datetime import datetime
from typing import Tuple, Type, List

# humanoid-gym
from humanoid_gym import LEGGED_GYM_ROOT_DIR
from humanoid_gym.utils.base_config import BaseConfig
from humanoid_gym.utils.helpers import (
    get_args,
    update_cfg_from_args,
    class_to_dict,
    get_load_path,
    set_seed,
    parse_sim_params,
)

# learning
from learning.env import VecEnv
from learning.runners import FLDOnPolicyRunner


class TaskRegistry:
    """This class simplifies creation of environments and agents."""

    def __init__(self):
        self.task_classes = dict()
        self.env_cfgs = dict()
        self.train_cfgs = dict()

    """
    Properties
    """

    def get_task_names(self) -> List[str]:
        """Returns a list of registered task names.

        Returns:
            List[str]: List of registered task names.
        """
        return list(self.task_classes.keys())

    def get_task_class(self, name: str) -> Type[VecEnv]:
        """Retrieve the class object corresponding to input name.

        Args:
            name (str): name of the registered environment.

        Raises:
            ValueError: When there is no registered environment with input `name`.

        Returns:
            Type[VecEnv]: The environment class object.
        """
        # check if there is a registered env with that name
        if self._check_valid_task(name):
            return self.task_classes[name]

    def get_cfgs(self, name: str) -> Tuple[BaseConfig, BaseConfig]:
        """Retrieve the default environment and training configurations.

        Args:
            name (str): Name of the environment.

         Raises:
            ValueError: When there is no registered environment with input `name`.

        Returns:
            Tuple[BaseConfig, BaseConfig]: The default environment and training configurations.
        """
        # check if there is a registered env with that name
        if self._check_valid_task(name):
            # retrieve configurations
            train_cfg = self.train_cfgs[name]
            env_cfg = self.env_cfgs[name]
            # copy seed between environment and agent
            env_cfg.seed = train_cfg.seed
            return env_cfg, train_cfg

    """
    Operations
    """

    def register(
        self,
        name: str,
        task_class: Type[VecEnv],
        env_cfg: Type[BaseConfig],
        train_cfg: Type[BaseConfig],
    ):
        """Add a particular environment to the task registry.

        Args:
            name (str): Name of the environment.
            task_class (Type[VecEnv]): The corresponding task class.
            env_cfg (Type[BaseConfig]): The corresponding environment configuration file.
            train_cfg (Type[BaseConfig]): The corresponding agent configuration file.
        """
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg()
        self.train_cfgs[name] = train_cfg()

    def make_env(
        self, name: str, args: argparse.Namespace = None, env_cfg: BaseConfig = None
    ) -> Tuple[VecEnv, BaseConfig]:
        """Creates an environment from the registry.

        Args:
            name (str): Name of a registered env.
            args (argparse.Namespace, optional): Parsed CLI arguments. If :obj:`None`, then
                `get_args()` is called to obtain arguments. Defaults to None.
            env_cfg (BaseConfig, optional): Environment configuration class instance used to
                overwrite the default registered configuration. Defaults to None.

        Raises:
            ValueError: When there is no registered environment with input `name`.

        Returns:
            Tuple[VecEnv, BaseConfig]: Tuple containing the created class instance and corresponding
                configuration class instance.
        """
        # check if there is a registered env with that name
        task_class = self.get_task_class(name)
        # if no args passed, get command line arguments
        if args is None:
            args = get_args()
        # if no config passed, use default env config
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        # set seed
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        # create environment instance
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        return env, env_cfg

    def make_alg_runner(
        self,
        env: VecEnv,
        name: str = None,
        args: argparse.Namespace = None,
        train_cfg: BaseConfig = None,
        log_root: str = "default",
    ) -> Tuple[FLDOnPolicyRunner, BaseConfig]:
        """Creates the training algorithm either from a registered name or from the provided
        config file.

        TODO (@nrudin): Remove environment from within the algorithm.

        Note:
            The training/agent configuration is loaded from either "name" or "train_cfg". If both are
            passed then the default configuration via "name" is ignored.

        Args:
            env (VecEnv): The environment to train on.
            name (str, optional): The environment name to retrieve corresponding training configuration.
                Defaults to None.
            args (argparse.Namespace, optional): Parsed CLI arguments. If :obj:`None`, then
                `get_args()` is called to obtain arguments. Defaults to None.
            train_cfg (BaseConfig, optional): Instance of training configuration class. If
                :obj:`None`, then `name` is used to retrieve default training configuration.
                Defaults to None.
            log_root (str, optional): Logging directory for TensorBoard. Set to obj:`None` to avoid
                logging (such as during test-time). Logs are saved in the `<log_root>/<date_time>_<run_name>`
                directory. If "default", then `log_root` is set to
                "{LEGGED_GYM_ROOT_DIR}/logs/{train_cfg.runner.experiment_name}". Defaults to "default".

        Raises:
            ValueError: If neither "name" or "train_cfg" are provided for loading training configuration.

        Returns:
            Tuple[WASABIOnPolicyRunner, BaseConfig]: Tuple containing the training runner and configuration instances.
        """
        # if config files are passed use them, otherwise load default from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("No training configuration provided. Either 'name' or 'train_cfg' must not be None.")
            else:
                # load config files
                _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"Training configuration instance provided. Ignoring default configuration for 'name={name}'.")
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        # resolve logging
        if log_root is None:
            log_dir_path = None
        else:
            # default location for logs
            if log_root == "default":
                log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
            # log directory
            log_dir_path = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
            )
        # create training runner
        runner_class = eval(train_cfg.runner_class_name)
        train_cfg_dict = class_to_dict(train_cfg)
        runner = runner_class(env, train_cfg_dict, log_dir_path, device=args.rl_device)
        # save resume path before creating a new log_dir
        runner.add_git_repo_to_log(__file__)
        # save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(
                log_root,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)

        return runner, train_cfg

    """
    Private helpers.
    """

    def _check_valid_task(self, name: str) -> bool:
        """Checks if input task name is valid.

        Args:
            name (str): Name of the registered task.

        Raises:
            ValueError: When there is no registered environment with input `name`.

        Returns:
            bool: True if the task exists.
        """
        registered_tasks = self.get_task_names()
        if name not in registered_tasks:
            print(f"The task '{name}' is not registered. Please use one of the following: ")
            for name in registered_tasks:
                print(f"\t - {name}")
            raise ValueError(f"[TaskRegistry]: Task with name: {name} is not registered.")
        else:
            return True


# make global task registry
task_registry = TaskRegistry()
