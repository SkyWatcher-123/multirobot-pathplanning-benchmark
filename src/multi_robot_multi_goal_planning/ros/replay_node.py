"""ROS1 node: replay a previously exported trajectory in MoveIt RViz.

Loads a DisplayTrajectory JSON produced by :mod:`plan_node` (or
``env.export_display_trajectory``) and republishes it on
``/move_group/display_planned_path`` so a plan can be reviewed in RViz without
re-running the planner. This is what makes a computed plan "replayable".
"""

from __future__ import annotations

import os


def main():
    import rospy

    rospy.init_node("mrmg_replay_node")

    from multi_robot_multi_goal_planning.ros.trajectory_conversion import (
        load_trajectory_json,
    )
    from multi_robot_multi_goal_planning.ros.moveit_interface import (
        DisplayTrajectoryPublisher,
    )

    filename = rospy.get_param("~file", "")
    loop = bool(rospy.get_param("~loop", True))
    rate_pad = float(rospy.get_param("~loop_pad", 1.0))

    if not filename:
        rospy.logfatal("[mrmg] set ~file to a trajectory JSON path")
        return

    filename = os.path.expanduser(filename)
    if not os.path.exists(filename):
        rospy.logfatal("[mrmg] trajectory file not found: %s", filename)
        return

    traj = load_trajectory_json(filename)
    pub = DisplayTrajectoryPublisher()
    duration = pub.replay_duration(traj)
    rospy.loginfo("[mrmg] replaying %s (%.1fs per loop)", filename, duration)

    while not rospy.is_shutdown():
        pub.publish_dict(traj)
        rospy.sleep(max(1.0, duration + rate_pad))
        if not loop:
            break

    rospy.loginfo("[mrmg] replay done")


if __name__ == "__main__":  # pragma: no cover
    main()
