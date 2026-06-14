"""Convert a planned path into a MoveIt-replayable trajectory.

The planners in this repository return a ``List[State]``, where each
:class:`~multi_robot_multi_goal_planning.problems.planning_env.State` holds a
multi-robot :class:`Configuration` and a :class:`Mode`. To replay such a path in
MoveIt RViz we turn it into a ``moveit_msgs/DisplayTrajectory`` published on
``/move_group/display_planned_path`` (the topic the MoveIt RViz "Trajectory" /
"MotionPlanning" displays listen to).

This module is deliberately free of any ``rospy``/``moveit_msgs`` import at module
scope: the conversion produces a plain, JSON-serialisable dictionary that mirrors
the ``DisplayTrajectory`` message structure. :func:`build_display_trajectory_msg`
then (lazily) turns that dictionary into the real ROS message when ROS is
available. This keeps the conversion logic unit-testable without a ROS install
and lets the path be exported to disk and replayed offline later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# We only need the lightweight pieces of the planning core here. Importing the
# submodules directly (rather than the package) avoids pulling in optional
# backends.
from multi_robot_multi_goal_planning.problems.planning_env import State, Mode


@dataclass
class TimedJointTrajectory:
    """A time-parameterised joint-space trajectory for a single (composite) model.

    ``joint_names`` lists every actuated joint of the *whole* scene (all robots
    concatenated, in the same order the planner concatenates the configuration).
    Each entry of ``points`` is ``(positions, time_from_start)`` where
    ``positions`` aligns with ``joint_names`` and ``time_from_start`` is seconds
    from the start of the trajectory.
    """

    joint_names: List[str]
    points: List[Tuple[List[float], float]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def duration(self) -> float:
        return self.points[-1][1] if self.points else 0.0


def _max_joint_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute per-joint difference (the ``max`` distance metric)."""
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def path_to_timed_trajectory(
    path: Sequence[State],
    joint_names: Sequence[str],
    velocity: float = 0.25,
    min_dt: float = 1e-3,
    start_time: float = 0.0,
) -> TimedJointTrajectory:
    """Time-parameterise a planned path in joint space.

    The time between consecutive waypoints is ``max(min_dt, delta / velocity)``
    where ``delta`` is the largest absolute joint motion between them (a constant
    max-joint-speed parameterisation). This mirrors the ``adapt_to_max_distance``
    behaviour the benchmark uses when animating paths, and produces a strictly
    increasing time stamp sequence that MoveIt/RViz requires.

    Parameters
    ----------
    path:
        The planned states. ``path[i].q.state()`` must be ordered consistently
        with ``joint_names``.
    joint_names:
        Names of every actuated joint, in the configuration's concatenation
        order.
    velocity:
        Nominal max joint speed (rad/s or m/s) used for time scaling.
    min_dt:
        Minimum time increment between waypoints (keeps timestamps strictly
        increasing even for tiny/zero moves).
    start_time:
        Time stamp of the first waypoint.
    """
    if velocity <= 0:
        raise ValueError("velocity must be positive")
    if len(path) == 0:
        raise ValueError("cannot convert an empty path")

    joint_names = list(joint_names)
    states = [np.asarray(s.q.state(), dtype=float) for s in path]

    n_dof = states[0].shape[0]
    if n_dof != len(joint_names):
        raise ValueError(
            f"configuration dimension ({n_dof}) does not match number of joint "
            f"names ({len(joint_names)})"
        )

    points: List[Tuple[List[float], float]] = []
    t = float(start_time)
    points.append((states[0].tolist(), t))
    for i in range(1, len(states)):
        dt = max(min_dt, _max_joint_delta(states[i], states[i - 1]) / velocity)
        t += dt
        points.append((states[i].tolist(), t))

    return TimedJointTrajectory(joint_names=joint_names, points=points)


def segment_path_by_mode(path: Sequence[State]) -> List[Tuple[int, int, Mode]]:
    """Split a path into contiguous same-mode segments.

    Returns a list of ``(start_index, end_index_inclusive, mode)`` tuples. The
    segments overlap by one index at mode boundaries so that, when emitted as
    separate ``RobotTrajectory`` entries, each segment starts exactly where the
    previous one ended (continuous replay across modes).
    """
    if len(path) == 0:
        return []

    segments: List[Tuple[int, int, Mode]] = []
    seg_start = 0
    for i in range(1, len(path)):
        if path[i].mode is not path[seg_start].mode and path[i].mode != path[seg_start].mode:
            segments.append((seg_start, i, path[seg_start].mode))
            seg_start = i
    segments.append((seg_start, len(path) - 1, path[seg_start].mode))
    return segments


def _attachment_to_dict(att) -> Dict[str, Any]:
    """Serialise an Attachment (from moveit_env) into a plain JSON-able dict."""
    return {
        "object_id": att.object_id,
        "link": att.link,
        "pose": list(att.pose),
        "mesh": att.mesh,
        "scale": list(att.scale),
        "touch_links": list(att.touch_links),
        "held": bool(att.held),
    }


def to_display_trajectory_dict(
    path: Sequence[State],
    joint_names: Sequence[str],
    base_frame: str = "world",
    velocity: float = 0.25,
    min_dt: float = 1e-3,
    split_by_mode: bool = True,
    model_id: str = "",
    extra_metadata: Optional[Dict[str, Any]] = None,
    attachments_provider=None,
) -> Dict[str, Any]:
    """Build a JSON-serialisable dict mirroring ``moveit_msgs/DisplayTrajectory``.

    The structure is::

        {
          "model_id": "",
          "trajectory_start": {"joint_state": {"name": [...], "position": [...]}},
          "trajectory": [
              {"joint_trajectory": {
                  "header": {"frame_id": base_frame},
                  "joint_names": [...],
                  "points": [{"positions": [...], "time_from_start": <sec>}, ...],
              }},
              ...   # one entry per mode segment when split_by_mode is True
          ],
          "metadata": {...},   # task ids per segment etc. (not part of the ROS msg)
        }

    When ``split_by_mode`` is True, each contiguous mode produces its own
    ``RobotTrajectory`` so RViz replays the multi-modal plan segment by segment
    (and the per-segment task ids are recorded in ``metadata``).
    """
    joint_names = list(joint_names)
    if len(path) == 0:
        raise ValueError("cannot convert an empty path")

    if split_by_mode:
        segments = segment_path_by_mode(path)
    else:
        segments = [(0, len(path) - 1, path[0].mode)]

    robot_trajectories: List[Dict[str, Any]] = []
    segment_meta: List[Dict[str, Any]] = []
    segment_attachments: List[List[Dict[str, Any]]] = []
    for (s, e, mode) in segments:
        sub_path = list(path[s : e + 1])
        timed = path_to_timed_trajectory(
            sub_path, joint_names, velocity=velocity, min_dt=min_dt
        )
        points = [
            {"positions": pos, "velocities": [], "accelerations": [],
             "effort": [], "time_from_start": tfs}
            for (pos, tfs) in timed.points
        ]
        robot_trajectories.append(
            {
                "joint_trajectory": {
                    "header": {"frame_id": base_frame},
                    "joint_names": joint_names,
                    "points": points,
                },
                "multi_dof_joint_trajectory": {
                    "header": {"frame_id": base_frame},
                    "joint_names": [],
                    "points": [],
                },
            }
        )
        attached = (
            [_attachment_to_dict(a) for a in attachments_provider(mode)]
            if attachments_provider is not None
            else []
        )
        segment_attachments.append(attached)
        segment_meta.append(
            {
                "task_ids": list(mode.task_ids) if mode is not None else None,
                "num_points": len(points),
                "duration": timed.duration,
                "attached_collision_objects": attached,
            }
        )

    first_positions = np.asarray(path[0].q.state(), dtype=float).tolist()
    result: Dict[str, Any] = {
        "model_id": model_id,
        "trajectory_start": {
            "joint_state": {
                "header": {"frame_id": base_frame},
                "name": joint_names,
                "position": first_positions,
                "velocity": [],
                "effort": [],
            },
            "multi_dof_joint_state": {},
            # Objects attached at the start of the trajectory (the first mode).
            "attached_collision_objects": segment_attachments[0] if segment_attachments else [],
            "is_diff": False,
        },
        "trajectory": robot_trajectories,
        "metadata": {
            "base_frame": base_frame,
            "velocity": velocity,
            "num_segments": len(robot_trajectories),
            "segments": segment_meta,
        },
    }
    if extra_metadata:
        result["metadata"].update(extra_metadata)
    return result


def save_trajectory_json(traj_dict: Dict[str, Any], filename: str) -> None:
    """Persist a display-trajectory dict so it can be replayed offline."""
    with open(filename, "w") as f:
        json.dump(traj_dict, f, indent=2)


def load_trajectory_json(filename: str) -> Dict[str, Any]:
    """Load a display-trajectory dict previously written by :func:`save_trajectory_json`."""
    with open(filename, "r") as f:
        return json.load(f)


def build_display_trajectory_msg(traj_dict: Dict[str, Any], attachment_builder=None):
    """Convert a display-trajectory dict into a ``moveit_msgs/DisplayTrajectory``.

    Imported lazily: ``moveit_msgs``/``trajectory_msgs``/``sensor_msgs`` are only
    needed at this point, so the rest of this module works without a ROS install.

    ``attachment_builder`` (optional) is a callable turning an attachment dict
    into a ``moveit_msgs/AttachedCollisionObject`` (see
    ``moveit_interface.build_attached_collision_object``); when given, the start
    state's ``attached_collision_objects`` are populated so grasped meshes show in
    RViz.
    """
    try:
        from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory, RobotState
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from sensor_msgs.msg import JointState
        import rospy
    except ImportError as err:  # pragma: no cover - requires ROS
        raise ImportError(
            "build_display_trajectory_msg requires a ROS1 environment with "
            "moveit_msgs, trajectory_msgs and sensor_msgs available. Source your "
            "catkin workspace before calling it."
        ) from err

    def _duration(seconds: float):
        return rospy.Duration.from_sec(float(seconds))

    msg = DisplayTrajectory()
    msg.model_id = traj_dict.get("model_id", "")

    start = traj_dict["trajectory_start"]
    rs = RobotState()
    js = JointState()
    js.name = list(start["joint_state"]["name"])
    js.position = list(start["joint_state"]["position"])
    rs.joint_state = js
    rs.is_diff = bool(start.get("is_diff", False))
    attached = start.get("attached_collision_objects", []) or []
    if attached and attachment_builder is not None:
        rs.attached_collision_objects = [attachment_builder(a) for a in attached]
        rs.is_diff = True
    msg.trajectory_start = rs

    for rt_dict in traj_dict["trajectory"]:
        rt = RobotTrajectory()
        jt = JointTrajectory()
        jt.header.frame_id = rt_dict["joint_trajectory"]["header"]["frame_id"]
        jt.joint_names = list(rt_dict["joint_trajectory"]["joint_names"])
        for p in rt_dict["joint_trajectory"]["points"]:
            pt = JointTrajectoryPoint()
            pt.positions = list(p["positions"])
            pt.velocities = list(p.get("velocities", []) or [])
            pt.accelerations = list(p.get("accelerations", []) or [])
            pt.time_from_start = _duration(p["time_from_start"])
            jt.points.append(pt)
        rt.joint_trajectory = jt
        msg.trajectory.append(rt)

    return msg
