import os

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
"""Absolute path to the solo-gym repository."""

LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "solo_gym", "envs")
"""Absolute path to the module `solo_gym.envs` in solo-gym repository."""
