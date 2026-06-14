import os
import sys
import importlib.util

# The rai (`robotic`) backend is an optional dependency, just like pinocchio and
# mujoco below. Guarding the import keeps the registry usable in environments
# where rai is not installed (e.g. a ROS1 Noetic / MoveIt deployment, which only
# needs the MoveIt backend). The concrete rai environments simply do not get
# registered when `robotic` is unavailable.
if importlib.util.find_spec("robotic") is not None:
    from . import rai_envs
    from . import rai_single_goal_envs
    from . import rai_unordered_envs
    from . import rai_free_envs
    from . import rai_envs_constrained

from . import abstract_env


if importlib.util.find_spec("pinocchio") is not None:
    from . import pinocchio_env

if importlib.util.find_spec("mujoco") is not None:
    from . import mujoco_env

# The MoveIt backend only needs `rospy`/`moveit_commander` at runtime when the
# collision checker actually talks to a live `move_group`. The environment module
# itself is import-safe without ROS, so we always register it.
from . import moveit_env

from .registry import get_env_by_name, get_all_environments

__all__ = ["get_env_by_name", "get_all_environments"]
