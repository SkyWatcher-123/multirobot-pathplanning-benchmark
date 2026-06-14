"""ROS1 / MoveIt integration for the multi-robot-multi-goal planning benchmark.

This subpackage bridges the backend-agnostic planning core (``problems`` and
``planners``) to a ROS1 Noetic + MoveIt deployment:

* :mod:`trajectory_conversion` -- turn a planned ``List[State]`` into a
  ``moveit_msgs/DisplayTrajectory`` (or a JSON-serialisable equivalent) so it can
  be replayed in MoveIt RViz. This module never imports ``rospy`` and is fully
  unit-testable without a ROS installation.
* :mod:`moveit_interface` -- thin wrapper around ``rospy`` / ``moveit_commander``
  providing the planning-scene (mesh collision objects), the
  ``/check_state_validity`` collision checker, and trajectory publishing. Imports
  ROS lazily so the rest of the package keeps working without ROS.

The actual MoveIt-backed planning problem lives in
:mod:`multi_robot_multi_goal_planning.problems.moveit_env`.
"""

from .trajectory_conversion import (
    TimedJointTrajectory,
    path_to_timed_trajectory,
    segment_path_by_mode,
    to_display_trajectory_dict,
    save_trajectory_json,
    load_trajectory_json,
    build_display_trajectory_msg,
)

__all__ = [
    "TimedJointTrajectory",
    "path_to_timed_trajectory",
    "segment_path_by_mode",
    "to_display_trajectory_dict",
    "save_trajectory_json",
    "load_trajectory_json",
    "build_display_trajectory_msg",
]
