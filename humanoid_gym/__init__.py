import os

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
"""Absolute path to the humanoid-gym repository."""

LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "humanoid_gym", "envs")
"""Absolute path to the module `humanoid_gym.envs` in humanoid-gym repository."""
