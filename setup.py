## Catkin Python setup for the `mrmg_moveit_planning` ROS package.
##
## This file is used by `catkin_python_setup()` to expose the
## `multi_robot_multi_goal_planning` Python package (under `src/`) inside a catkin
## workspace. Normal (non-ROS) pip installs use pyproject.toml instead -- pip
## prefers the PEP 517 build backend declared there, so the two coexist.

from setuptools import find_packages
from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={
        # Ship the bundled MoveIt demo problem spec so the registered demo
        # environments load in an install space as well as a devel space.
        "multi_robot_multi_goal_planning": ["assets/desc/moveit_*.json"],
    },
)

setup(**setup_args)
