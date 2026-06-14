"""Live ROS1 / MoveIt plumbing for the MoveIt planning backend.

This module is the only place that touches ``rospy`` / ``moveit_commander`` /
``moveit_msgs``. It provides:

* :class:`MoveItSceneManager` -- add the (mesh) collision objects of a problem to
  the MoveIt planning scene so they participate in collision checking and show up
  in RViz;
* :class:`MoveItServiceCollisionChecker` -- a
  :class:`~multi_robot_multi_goal_planning.problems.moveit_env.StateValidityChecker`
  that delegates to ``move_group``'s ``/check_state_validity``
  (``moveit_msgs/GetStateValidity``) service;
* :class:`DisplayTrajectoryPublisher` -- publish a planned path as a
  ``moveit_msgs/DisplayTrajectory`` on ``/move_group/display_planned_path`` for
  RViz replay;
* :func:`attach_moveit_runtime` -- wire all of the above into a
  :class:`MoveItEnvironment`.

ROS imports are performed lazily so that simply importing this module (e.g. for
documentation or unit-test discovery) does not require a ROS installation;
constructing the classes does.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.moveit_env import (
    MoveItEnvironment,
    StateValidityChecker,
)
from multi_robot_multi_goal_planning.problems.moveit_problem_spec import MeshObject

try:  # pragma: no cover - requires a ROS install
    import rospy
    import moveit_commander
    from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
    from moveit_msgs.msg import (
        RobotState,
        AttachedCollisionObject,
        CollisionObject,
        DisplayTrajectory,
    )
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import Pose, PoseStamped, Point
    from shape_msgs.msg import Mesh, MeshTriangle

    HAVE_ROS = True
except ImportError:  # pragma: no cover
    HAVE_ROS = False


# --------------------------------------------------------------------------- #
#  Mesh / attached-object helpers (for grasped movable objects)
# --------------------------------------------------------------------------- #
_MESH_MSG_CACHE: dict = {}


def make_mesh_msg(filename: str, scale=(1.0, 1.0, 1.0)):
    """Load a mesh file into a ``shape_msgs/Mesh`` (cached). Requires pyassimp."""
    _require_ros()
    key = (filename, tuple(scale))
    if key in _MESH_MSG_CACHE:
        return _MESH_MSG_CACHE[key]
    try:
        import pyassimp  # type: ignore
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "pyassimp is required to attach mesh objects (grasped meshes). "
            "Install python3-pyassimp."
        ) from err

    scene = pyassimp.load(filename)
    try:
        if not scene.meshes:
            raise ValueError(f"no meshes found in {filename}")
        m = scene.meshes[0]
        mesh = Mesh()
        for face in m.faces:
            idx = list(face) if not hasattr(face, "indices") else list(face.indices)
            if len(idx) == 3:
                tri = MeshTriangle()
                tri.vertex_indices = [int(idx[0]), int(idx[1]), int(idx[2])]
                mesh.triangles.append(tri)
        for v in m.vertices:
            mesh.vertices.append(
                Point(x=float(v[0]) * scale[0], y=float(v[1]) * scale[1], z=float(v[2]) * scale[2])
            )
    finally:
        pyassimp.release(scene)

    _MESH_MSG_CACHE[key] = mesh
    return mesh


def _pose_from7(pose7):
    p = Pose()
    p.position.x, p.position.y, p.position.z = pose7[0:3]
    (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w) = pose7[3:7]
    return p


def build_attached_collision_object(att):
    """Build a ``moveit_msgs/AttachedCollisionObject`` from an env Attachment.

    ``att`` is a ``multi_robot_multi_goal_planning.problems.moveit_env.Attachment``
    (or the JSON dict form with the same fields). The mesh is resolved + loaded
    (cached); the object is parented to ``att.link`` with its relative pose.
    """
    _require_ros()
    from multi_robot_multi_goal_planning.ros.mesh_resolver import resolve_mesh_path

    object_id = att["object_id"] if isinstance(att, dict) else att.object_id
    link = att["link"] if isinstance(att, dict) else att.link
    pose = att["pose"] if isinstance(att, dict) else att.pose
    mesh = att["mesh"] if isinstance(att, dict) else att.mesh
    scale = att["scale"] if isinstance(att, dict) else att.scale
    touch_links = att["touch_links"] if isinstance(att, dict) else att.touch_links

    mesh_msg = make_mesh_msg(resolve_mesh_path(mesh), tuple(scale))

    co = CollisionObject()
    co.id = object_id
    co.header.frame_id = link
    co.meshes = [mesh_msg]
    co.mesh_poses = [_pose_from7(pose)]
    co.operation = CollisionObject.ADD

    aco = AttachedCollisionObject()
    aco.link_name = link
    aco.object = co
    aco.touch_links = list(touch_links)
    return aco


def _require_ros() -> None:
    if not HAVE_ROS:
        raise ImportError(
            "ROS1 (rospy, moveit_commander, moveit_msgs) is not available. Source "
            "your catkin workspace (e.g. `source /opt/ros/noetic/setup.bash` and "
            "your workspace's devel/setup.bash) before using the MoveIt runtime."
        )


# --------------------------------------------------------------------------- #
#  Planning scene (mesh collision objects)
# --------------------------------------------------------------------------- #
class MoveItSceneManager:
    """Manage the MoveIt planning scene for a problem (mesh collision objects)."""

    def __init__(self, base_frame: str = "world", wait: float = 2.0):
        _require_ros()
        self.base_frame = base_frame
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        # Give the scene monitor a moment to connect.
        rospy.sleep(wait)
        self._added: List[str] = []

    @staticmethod
    def _pose(obj: MeshObject) -> "PoseStamped":
        ps = PoseStamped()
        ps.header.frame_id = obj.frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = obj.position
        (
            ps.pose.orientation.x,
            ps.pose.orientation.y,
            ps.pose.orientation.z,
            ps.pose.orientation.w,
        ) = obj.orientation
        return ps

    def add_mesh_objects(self, objects: List[MeshObject]) -> None:
        """Add each mesh collision object to the planning scene.

        Assumption (per the problem spec): every planning-scene object is a mesh,
        which is the representation MoveIt collision checking expects.
        """
        from multi_robot_multi_goal_planning.ros.mesh_resolver import resolve_mesh_path

        for obj in objects:
            ps = self._pose(obj)
            filename = resolve_mesh_path(obj.mesh)
            self.scene.add_mesh(obj.id, ps, filename, size=obj.scale)
            self._added.append(obj.id)
            rospy.loginfo("[mrmg] added mesh collision object '%s' (%s)", obj.id, filename)
        self._wait_for_objects(self._added)

    def _wait_for_objects(self, names: List[str], timeout: float = 5.0) -> None:
        start = rospy.get_time()
        while rospy.get_time() - start < timeout and not rospy.is_shutdown():
            known = set(self.scene.get_known_object_names())
            if all(n in known for n in names):
                return
            rospy.sleep(0.1)
        missing = [n for n in names if n not in set(self.scene.get_known_object_names())]
        if missing:
            rospy.logwarn("[mrmg] planning-scene objects not confirmed: %s", missing)

    def clear(self) -> None:
        for name in self._added:
            self.scene.remove_world_object(name)
        self._added = []


# --------------------------------------------------------------------------- #
#  Collision checking via /check_state_validity
# --------------------------------------------------------------------------- #
class MoveItServiceCollisionChecker(StateValidityChecker):
    """State-validity checker backed by MoveIt's ``/check_state_validity`` service.

    Parameters
    ----------
    group_name:
        Planning group to validate against. The default ``""`` checks the whole
        robot model (all robots) against itself and the world -- which is what we
        want for multi-robot collision checking. Provide a combined SRDF group if
        your MoveIt setup requires a non-empty group.
    service:
        Service name (default ``/check_state_validity``).
    """

    def __init__(
        self,
        group_name: str = "",
        service: str = "/check_state_validity",
        wait: bool = True,
    ):
        _require_ros()
        self.group_name = group_name
        self.service_name = service
        if wait:
            rospy.loginfo("[mrmg] waiting for %s ...", service)
            rospy.wait_for_service(service)
        self._srv = rospy.ServiceProxy(service, GetStateValidity, persistent=True)
        # Optional hook: callable(mode) -> List[AttachedCollisionObject]
        self.attached_objects_for_mode = None
        self._attachments_provider = None
        self._aco_cache: dict = {}

    def enable_attachments(self, attachments_provider) -> None:
        """Represent movable (graspable) objects as attached collision objects.

        ``attachments_provider`` is ``env.attachments_for_mode``: a callable
        ``mode -> List[Attachment]`` describing which movable mesh is parented to
        which link (world anchor when resting, gripper link when held) in that
        mode. Each is turned into an ``AttachedCollisionObject`` added to the
        per-check ``RobotState`` (cached by object/link/pose), so the carried mesh
        participates in collision checking and the world copy never double-counts.
        """
        self._attachments_provider = attachments_provider
        self.attached_objects_for_mode = self._build_attachments

    def _build_attachments(self, mode):
        if self._attachments_provider is None:
            return []
        out = []
        for att in self._attachments_provider(mode):
            key = (att.object_id, att.link, tuple(att.pose))
            aco = self._aco_cache.get(key)
            if aco is None:
                aco = build_attached_collision_object(att)
                self._aco_cache[key] = aco
            out.append(aco)
        return out

    def is_valid(
        self, joint_names: List[str], positions: NDArray, mode
    ) -> bool:
        rs = RobotState()
        js = JointState()
        js.name = list(joint_names)
        js.position = list(np.asarray(positions, dtype=float))
        rs.joint_state = js

        if self.attached_objects_for_mode is not None and mode is not None:
            attached = self.attached_objects_for_mode(mode)
            if attached:
                rs.attached_collision_objects = list(attached)
                rs.is_diff = True

        req = GetStateValidityRequest()
        req.robot_state = rs
        req.group_name = self.group_name
        try:
            res = self._srv(req)
        except rospy.ServiceException as err:  # pragma: no cover
            rospy.logerr_throttle(5.0, "[mrmg] state validity service failed: %s" % err)
            return False
        return bool(res.valid)


# --------------------------------------------------------------------------- #
#  Trajectory publishing (RViz replay)
# --------------------------------------------------------------------------- #
class DisplayTrajectoryPublisher:
    """Publish a planned path as ``moveit_msgs/DisplayTrajectory`` for RViz."""

    def __init__(
        self,
        topic: str = "/move_group/display_planned_path",
        queue_size: int = 1,
        latch: bool = True,
    ):
        _require_ros()
        self.pub = rospy.Publisher(
            topic, DisplayTrajectory, queue_size=queue_size, latch=latch
        )
        # Give RViz a moment to subscribe to a latched publisher.
        rospy.sleep(0.5)

    def publish_dict(self, traj_dict: Dict) -> "DisplayTrajectory":
        from multi_robot_multi_goal_planning.ros.trajectory_conversion import (
            build_display_trajectory_msg,
        )

        msg = build_display_trajectory_msg(
            traj_dict, attachment_builder=build_attached_collision_object
        )
        self.pub.publish(msg)
        return msg

    def publish_segments(self, traj_dict: Dict, pause_pad: float = 1.0) -> None:
        """Replay a manipulation plan segment by segment.

        A single DisplayTrajectory shares one start state, so a grasped object
        could not change attachment mid-trajectory. For manipulation plans we
        therefore publish one DisplayTrajectory per mode segment, each carrying
        that segment's attached objects in ``trajectory_start`` -- so the carried
        mesh follows the gripper and detaches on place in RViz.
        """
        from multi_robot_multi_goal_planning.ros.trajectory_conversion import (
            build_display_trajectory_msg,
        )

        segments_meta = traj_dict.get("metadata", {}).get("segments", [])
        for i, rt in enumerate(traj_dict.get("trajectory", [])):
            if rospy.is_shutdown():
                return
            attached = segments_meta[i].get("attached_collision_objects", []) if i < len(segments_meta) else []
            seg_dict = {
                "model_id": traj_dict.get("model_id", ""),
                "trajectory_start": {
                    "joint_state": {
                        "name": rt["joint_trajectory"]["joint_names"],
                        "position": rt["joint_trajectory"]["points"][0]["positions"],
                    },
                    "attached_collision_objects": attached,
                    "is_diff": True,
                },
                "trajectory": [rt],
            }
            msg = build_display_trajectory_msg(
                seg_dict, attachment_builder=build_attached_collision_object
            )
            self.pub.publish(msg)
            pts = rt["joint_trajectory"]["points"]
            dur = float(pts[-1]["time_from_start"]) if pts else 1.0
            rospy.sleep(max(1.0, dur + pause_pad))

    def replay_duration(self, traj_dict: Dict) -> float:
        total = 0.0
        for rt in traj_dict.get("trajectory", []):
            pts = rt["joint_trajectory"]["points"]
            if pts:
                total += float(pts[-1]["time_from_start"])
        return total


# --------------------------------------------------------------------------- #
#  Wiring helper
# --------------------------------------------------------------------------- #
def read_joint_limits_from_param(joint_names: List[str]) -> Optional[Dict[str, tuple]]:
    """Read joint limits from the URDF on the parameter server (best effort)."""
    _require_ros()
    try:
        from urdf_parser_py.urdf import URDF  # type: ignore
    except ImportError:
        rospy.logwarn("[mrmg] urdf_parser_py not available; cannot read URDF limits")
        return None
    try:
        robot = URDF.from_parameter_server()
    except Exception as err:  # pragma: no cover
        rospy.logwarn("[mrmg] could not parse robot_description: %s", err)
        return None

    limits: Dict[str, tuple] = {}
    for joint in robot.joints:
        if joint.name in joint_names and joint.limit is not None:
            limits[joint.name] = (joint.limit.lower, joint.limit.upper)
    missing = [n for n in joint_names if n not in limits]
    if missing:
        rospy.logwarn("[mrmg] no URDF limits found for joints: %s", missing)
        return None
    return limits


def attach_moveit_runtime(
    env: MoveItEnvironment,
    check_group: str = "",
    add_scene_objects: bool = True,
    use_urdf_limits: bool = True,
) -> Dict:
    """Wire the live MoveIt runtime (scene + checker + RViz publisher) into ``env``.

    Returns a dict with the created ``scene``, ``checker`` and ``publisher`` so the
    caller can keep references / clean up.
    """
    _require_ros()

    runtime: Dict = {}

    # Only *static* objects go into the world planning scene. Movable (graspable)
    # objects are represented per-mode as attached collision objects on the
    # checker, so they are never double-counted as both world geometry and a
    # grasped object.
    static_objects = [o for o in env.collision_objects if not getattr(o, "movable", False)]
    if add_scene_objects and static_objects:
        scene = MoveItSceneManager(base_frame=env.base_frame)
        scene.add_mesh_objects(static_objects)
        runtime["scene"] = scene

    checker = MoveItServiceCollisionChecker(group_name=check_group)
    if getattr(env, "manipulating_env", False):
        checker.enable_attachments(env.attachments_for_mode)
        rospy.loginfo("[mrmg] attach/detach enabled for movable objects: %s",
                      sorted(getattr(env, "movable_objects", set())))
    env.set_validity_checker(checker)
    runtime["checker"] = checker

    if use_urdf_limits:
        limits = read_joint_limits_from_param(env.joint_names)
        if limits is not None:
            env.set_limits_from_dict(limits)
            rospy.loginfo("[mrmg] joint sampling limits set from URDF")

    publisher = DisplayTrajectoryPublisher()
    manipulating = getattr(env, "manipulating_env", False)

    def _display(path):
        traj = env.to_display_trajectory(path)
        if manipulating:
            publisher.publish_segments(traj)
        else:
            publisher.publish_dict(traj)

    env.set_display_function(_display)
    runtime["publisher"] = publisher

    return runtime
