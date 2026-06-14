"""ROS1 node: build a MoveIt problem, run a benchmark planner, replay in RViz.

This is the end-to-end entry point for the ROS1 / MoveIt deployment:

1. build a :class:`MoveItEnvironment` from a problem spec (a provided dependency
   graph + robots + mesh objects) or a registered env name;
2. wire the live MoveIt runtime (planning scene with the mesh collision objects +
   the ``/check_state_validity`` collision checker + an RViz trajectory
   publisher);
3. run one of the repository's own planners (RRT* by default -- *not* MoveIt's
   OMPL pipeline);
4. publish the resulting multi-modal path as a ``moveit_msgs/DisplayTrajectory``
   on ``/move_group/display_planned_path`` so it animates in MoveIt RViz, and
   export it to JSON for offline replay.

Run via ``rosrun mrmg_moveit_planning mrmg_plan_node`` or the provided launch
files. All behaviour is controlled through private ROS parameters (see ``main``).
"""

from __future__ import annotations

import os
import random

import numpy as np


def _build_planner(name, env):
    """Instantiate one of the repository planners by name."""
    from multi_robot_multi_goal_planning.planners.planner_rrtstar import RRTstar
    from multi_robot_multi_goal_planning.planners.planner_birrtstar import (
        BidirectionalRRTstar,
    )
    from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
    from multi_robot_multi_goal_planning.planners.composite_prm_planner import (
        CompositePRM,
        CompositePRMConfig,
    )
    from multi_robot_multi_goal_planning.planners.planner_aitstar import (
        AITstar,
        BaseITConfig,
    )
    from multi_robot_multi_goal_planning.planners.planner_eitstar import EITstar

    name = (name or "rrt_star").lower()
    if name in ("rrt_star", "rrtstar", "rrt"):
        return RRTstar(env, BaseRRTConfig())
    if name in ("birrt_star", "birrtstar", "birrt"):
        return BidirectionalRRTstar(env, BaseRRTConfig())
    if name in ("composite_prm", "prm"):
        return CompositePRM(env, CompositePRMConfig())
    if name in ("aitstar", "ait"):
        return AITstar(env, BaseITConfig())
    if name in ("eitstar", "eit"):
        return EITstar(env, BaseITConfig())
    raise ValueError(f"unknown planner '{name}'")


def main():
    import rospy

    rospy.init_node("mrmg_plan_node")

    from multi_robot_multi_goal_planning.problems.moveit_env import (
        make_moveit_env_from_file,
    )
    from multi_robot_multi_goal_planning.problems import get_env_by_name
    from multi_robot_multi_goal_planning.problems.util import interpolate_path
    from multi_robot_multi_goal_planning.planners.termination_conditions import (
        RuntimeTerminationCondition,
        IterationTerminationCondition,
    )
    from multi_robot_multi_goal_planning.ros.moveit_interface import attach_moveit_runtime

    # ----------------------------------------------------------- parameters
    problem_spec = rospy.get_param("~problem_spec", "")
    env_name = rospy.get_param("~env_name", "")
    mode_logic = rospy.get_param("~mode_logic", "")  # override spec's mode_logic
    planner_name = rospy.get_param("~planner", "rrt_star")
    max_time = rospy.get_param("~max_time", 10.0)
    num_iters = rospy.get_param("~num_iters", 0)
    seed = int(rospy.get_param("~seed", 1))
    check_group = rospy.get_param("~check_group", "")
    optimize = bool(rospy.get_param("~optimize", False))
    do_shortcut = bool(rospy.get_param("~shortcut", True))
    interp_res = float(rospy.get_param("~interpolation_resolution", 0.05))
    output = rospy.get_param("~output", "")
    do_replay = bool(rospy.get_param("~replay", True))
    num_replays = int(rospy.get_param("~num_replays", 1))
    use_urdf_limits = bool(rospy.get_param("~use_urdf_limits", True))

    np.random.seed(seed)
    random.seed(seed)

    # ----------------------------------------------------------- build env
    if problem_spec:
        rospy.loginfo("[mrmg] loading problem spec from %s", problem_spec)
        env = make_moveit_env_from_file(problem_spec)
        if mode_logic:
            rospy.logwarn(
                "[mrmg] mode_logic override requested but spec already built; "
                "set mode_logic inside the spec file instead."
            )
    elif env_name:
        rospy.loginfo("[mrmg] building registered env '%s'", env_name)
        env = get_env_by_name(env_name)
    else:
        rospy.logfatal("[mrmg] set either ~problem_spec or ~env_name")
        return

    # ----------------------------------------------- wire live MoveIt runtime
    attach_moveit_runtime(
        env,
        check_group=check_group,
        add_scene_objects=True,
        use_urdf_limits=use_urdf_limits,
    )

    if not env.is_collision_free(env.start_pos, env.start_mode):
        rospy.logwarn(
            "[mrmg] start configuration reports a collision; check the planning "
            "scene / start state."
        )

    # ------------------------------------------------------------- plan
    planner = _build_planner(planner_name, env)
    if num_iters and num_iters > 0:
        ptc = IterationTerminationCondition(num_iters)
    else:
        ptc = RuntimeTerminationCondition(max_time)

    rospy.loginfo("[mrmg] planning with %s ...", planner_name)
    path, info = planner.plan(ptc=ptc, optimize=optimize)

    if path is None:
        rospy.logerr("[mrmg] planner did not find a solution")
        return

    rospy.loginfo(
        "[mrmg] solution found: %d waypoints, cost %s",
        len(path),
        info.get("costs", ["?"])[-1] if info.get("costs") else "?",
    )

    # ----------------------------------------------------- post-process
    if do_shortcut:
        try:
            from multi_robot_multi_goal_planning.planners.shortcutting import (
                robot_mode_shortcut,
            )

            path, _ = robot_mode_shortcut(
                env,
                path,
                250,
                tolerance=env.collision_tolerance,
                resolution=env.collision_resolution,
            )
            rospy.loginfo("[mrmg] shortcutting done (%d waypoints)", len(path))
        except Exception as err:  # pragma: no cover
            rospy.logwarn("[mrmg] shortcutting skipped: %s", err)

    display_path = interpolate_path(path, interp_res, kind="euclidean")

    if not env.is_valid_plan(display_path):
        rospy.logwarn("[mrmg] interpolated plan failed validation (see logs)")

    # ----------------------------------------------------- export + replay
    traj = env.to_display_trajectory(display_path, split_by_mode=True)

    if output:
        output = os.path.expanduser(output)
        env.export_display_trajectory(display_path, output, split_by_mode=True)
        rospy.loginfo("[mrmg] trajectory exported to %s", output)

    if do_replay:
        from multi_robot_multi_goal_planning.ros.moveit_interface import (
            DisplayTrajectoryPublisher,
        )

        pub = DisplayTrajectoryPublisher()
        duration = pub.replay_duration(traj)
        rospy.loginfo("[mrmg] replaying in RViz (%.1fs per loop)", duration)
        for i in range(max(1, num_replays)):
            if rospy.is_shutdown():
                break
            pub.publish_dict(traj)
            rospy.loginfo("[mrmg] replay %d/%d", i + 1, num_replays)
            rospy.sleep(max(1.0, duration + 1.0))

    rospy.loginfo("[mrmg] done")


if __name__ == "__main__":  # pragma: no cover
    main()
