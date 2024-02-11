##
# Locomotion environments.
##
# fmt: off
from .base.legged_robot import LeggedRobot
from .mit_humanoid.mit_humanoid import MITHumanoid
from .mit_humanoid.mit_humanoid_config import (
    MITHumanoidFlatCfg, 
    MITHumanoidFlatCfgPPO
)

# fmt: on

##
# Task registration
##
from humanoid_gym.utils.task_registry import task_registry

task_registry.register("mit_humanoid", MITHumanoid, MITHumanoidFlatCfg, MITHumanoidFlatCfgPPO)