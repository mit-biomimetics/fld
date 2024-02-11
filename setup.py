"""Installation script for the 'humanoid_gym' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "isaacgym",
    "matplotlib",
    "tensorboard",
    "torch>=1.4.0",
    "torchvision>=0.5.0",
    "onnx",
    "numpy>=1.16.4,<=1.22.4",
    "setuptools==59.5.0",
    "gym>=0.17.1",
    "GitPython",
    "scikit-learn>=1.2.1",
    "faiss-gpu>=1.7.2",
    "Pillow==9.5.0",
]

# Installation operation
setup(
    name="humanoid_gym",
    version="1.0.0",
    author="Chenhao Li",
    packages=find_packages(),
    author_email="chenhli@ethz.ch",
    description="Isaac Gym environments for MIT Humanoid",
    install_requires=INSTALL_REQUIRES,
)
