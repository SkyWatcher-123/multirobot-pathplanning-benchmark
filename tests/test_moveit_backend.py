"""Tests for the MoveIt backend that run without a ROS installation.

These cover everything that does not require a live ``move_group``:

* the problem-spec parser (robots / tasks / mesh objects / dependency graph),
* the MoveIt environment + an injected analytic collision checker,
* running a benchmark planner (RRT*) on a MoveIt problem,
* converting the planned multi-modal path into a MoveIt ``DisplayTrajectory``
  dict (the structure replayed in RViz) and a JSON round-trip,
* the ``package://`` mesh resolver and import-safety of the ROS glue.

The live-MoveIt path (``/check_state_validity`` etc.) is exercised on a real
Noetic + MoveIt machine via the launch files; it is intentionally not unit-tested
here.
"""

import json
import os

import numpy as np
import pytest

from multi_robot_multi_goal_planning.problems.moveit_problem_spec import (
    MoveItProblemSpec,
)
from multi_robot_multi_goal_planning.problems.moveit_env import (
    build_moveit_env,
    CallableStateValidityChecker,
)
from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.ros.trajectory_conversion import (
    path_to_timed_trajectory,
    segment_path_by_mode,
    to_display_trajectory_dict,
    save_trajectory_json,
    load_trajectory_json,
)
from multi_robot_multi_goal_planning.ros.mesh_resolver import resolve_mesh_path

from multi_robot_multi_goal_planning.planners.planner_rrtstar import RRTstar
from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)


SPEC = {
    "name": "unit_demo",
    "base_frame": "world",
    "velocity": 0.25,
    "mode_logic": "dependency",
    "collision": {"tolerance": 0.0, "resolution": 0.05},
    "robots": [
        {"name": "a1", "group": "arm_1", "joints": ["a1_x", "a1_y"],
         "start": [-0.8, 0.0], "limits": [[-2.0, -2.0], [2.0, 2.0]]},
        {"name": "a2", "group": "arm_2", "joints": ["a2_x", "a2_y"],
         "start": [0.8, 0.0], "limits": [[-2.0, -2.0], [2.0, 2.0]]},
    ],
    "collision_objects": [
        {"id": "box", "mesh": "package://pkg/meshes/box.stl", "frame": "world",
         "pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}},
    ],
    "tasks": [
        {"name": "a1_goal", "robots": ["a1"], "goal": {"type": "single", "config": [0.8, 0.6]}},
        {"name": "a2_goal_0", "robots": ["a2"], "goal": {"type": "single", "config": [0.0, 0.8]}},
        {"name": "a2_goal_1", "robots": ["a2"], "goal": {"type": "single", "config": [-0.8, 0.6]}},
        {"name": "terminal", "robots": ["a1", "a2"],
         "goal": {"type": "single", "config": [0.8, 0.6, -0.8, 0.6]}},
    ],
    "dependencies": [
        ["a2_goal_1", "a2_goal_0"],
        ["terminal", "a1_goal"],
        ["terminal", "a2_goal_1"],
    ],
}


def _analytic_checker():
    def fn(joint_names, positions, mode):
        p = np.asarray(positions, dtype=float)
        a, b = p[0:2], p[2:4]
        for pt in (a, b):
            if np.all(np.abs(pt) < 0.25):
                return False
        return np.linalg.norm(a - b) >= 0.2
    return CallableStateValidityChecker(fn)


# ------------------------------------------------------------------- spec
def test_spec_parses_robots_tasks_and_graph():
    spec = MoveItProblemSpec.from_dict(SPEC)
    assert spec.robot_names == ["a1", "a2"]
    assert spec.robot_dims == {"a1": 2, "a2": 2}
    assert spec.joint_names == ["a1_x", "a1_y", "a2_x", "a2_y"]
    np.testing.assert_allclose(spec.start_configuration().state(), [-0.8, 0, 0.8, 0])

    graph = spec.build_dependency_graph()
    assert graph.get_leaf_nodes() == {"terminal"}
    assert "a2_goal_0" in graph.get_all_dependencies("a2_goal_1")
    # one mesh object parsed
    assert len(spec.collision_objects) == 1
    assert spec.collision_objects[0].id == "box"


def test_spec_json_roundtrip():
    spec = MoveItProblemSpec.from_json(json.dumps(SPEC))
    assert spec.mode_logic == "dependency"
    assert len(spec.tasks) == 4


def test_spec_rejects_unknown_dependency():
    bad = json.loads(json.dumps(SPEC))
    bad["dependencies"].append(["terminal", "does_not_exist"])
    with pytest.raises(ValueError):
        MoveItProblemSpec.from_dict(bad).build_dependency_graph()


def test_spec_limits_optional():
    no_limits = json.loads(json.dumps(SPEC))
    for r in no_limits["robots"]:
        r.pop("limits")
    spec = MoveItProblemSpec.from_dict(no_limits)
    assert spec.limits() is None  # -> read from URDF at runtime


# ------------------------------------------------------------------- env
def test_env_builds_and_start_is_valid():
    spec = MoveItProblemSpec.from_dict(SPEC)
    env = build_moveit_env(spec, _analytic_checker())
    assert env.robots == ["a1", "a2"]
    assert env.is_collision_free(env.start_pos, env.start_mode)
    # central obstacle is rejected
    bad = env.start_pos.from_flat(np.array([0.0, 0.0, 1.5, 1.5]))
    assert not env.is_collision_free(bad, env.start_mode)


def test_env_requires_checker():
    spec = MoveItProblemSpec.from_dict(SPEC)
    env = build_moveit_env(spec)  # no checker injected
    with pytest.raises(RuntimeError):
        env.is_collision_free(env.start_pos, env.start_mode)


def test_registered_demo_envs_exist():
    env = get_env_by_name("moveit.demo_dependency")
    assert env.joint_names == ["a1_x", "a1_y", "a2_x", "a2_y"]
    assert get_env_by_name("moveit.demo_sequence") is not None


# ------------------------------------------------- planner + conversion
def _plan(env, seed=1, t=10.0):
    import random
    np.random.seed(seed)
    random.seed(seed)
    path, info = RRTstar(env, BaseRRTConfig()).plan(
        ptc=RuntimeTerminationCondition(t), optimize=False
    )
    return path, info


def test_planner_solves_moveit_dependency_problem():
    spec = MoveItProblemSpec.from_dict(SPEC)
    env = build_moveit_env(spec, _analytic_checker())
    path, _ = _plan(env)
    assert path is not None
    assert np.array_equal(path[0].q.state(), env.start_pos.state())
    assert env.is_terminal_mode(path[-1].mode)
    assert env.is_valid_plan(path)


def test_path_to_timed_trajectory_is_monotonic():
    spec = MoveItProblemSpec.from_dict(SPEC)
    env = build_moveit_env(spec, _analytic_checker())
    path, _ = _plan(env)
    timed = path_to_timed_trajectory(path, env.joint_names, velocity=0.25)
    assert timed.joint_names == env.joint_names
    times = [t for _, t in timed.points]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert len(timed.points[0][0]) == 4


def test_display_trajectory_dict_structure_and_modes():
    spec = MoveItProblemSpec.from_dict(SPEC)
    env = build_moveit_env(spec, _analytic_checker())
    path, _ = _plan(env)

    # The demo path passes through 4 modes -> 4 RobotTrajectory segments.
    n_segments = len(segment_path_by_mode(path))
    traj = to_display_trajectory_dict(path, env.joint_names, base_frame="world")

    assert traj["trajectory_start"]["joint_state"]["name"] == env.joint_names
    assert len(traj["trajectory"]) == n_segments == traj["metadata"]["num_segments"]
    for seg in traj["trajectory"]:
        assert seg["joint_trajectory"]["joint_names"] == env.joint_names
        pts = seg["joint_trajectory"]["points"]
        ts = [p["time_from_start"] for p in pts]
        assert all(b > a for a, b in zip(ts, ts[1:]))
    # first start position equals the planned start configuration
    np.testing.assert_allclose(
        traj["trajectory_start"]["joint_state"]["position"], env.start_pos.state()
    )


def test_trajectory_json_roundtrip(tmp_path):
    spec = MoveItProblemSpec.from_dict(SPEC)
    env = build_moveit_env(spec, _analytic_checker())
    path, _ = _plan(env)
    traj = to_display_trajectory_dict(path, env.joint_names)
    f = os.path.join(str(tmp_path), "traj.json")
    save_trajectory_json(traj, f)
    back = load_trajectory_json(f)
    assert back["metadata"]["num_segments"] == traj["metadata"]["num_segments"]
    assert back["trajectory"][0]["joint_trajectory"]["joint_names"] == env.joint_names


# ------------------------------------------------------------- ros glue
def test_mesh_resolver_passthrough_and_file_uri():
    assert resolve_mesh_path("/abs/path/mesh.stl") == "/abs/path/mesh.stl"
    assert resolve_mesh_path("file:///abs/path/mesh.stl") == "/abs/path/mesh.stl"


def test_ros_glue_imports_without_ros():
    # moveit_interface guards its ROS imports; importing it must not fail and
    # HAVE_ROS reflects whether ROS is present (False in this CI container).
    from multi_robot_multi_goal_planning.ros import moveit_interface as mi
    assert hasattr(mi, "HAVE_ROS")
    if not mi.HAVE_ROS:
        with pytest.raises(ImportError):
            mi.MoveItServiceCollisionChecker()


PICK_PLACE = {
    "name": "pp",
    "base_frame": "world",
    "anchor_link": "base_link",
    "mode_logic": "dependency",
    "robots": [
        {"name": "a1", "group": "arm_1", "joints": ["a1_x", "a1_y"], "start": [-0.8, 0.0],
         "limits": [[-2, -2], [2, 2]], "attach_link": "a1_tool",
         "touch_links": ["a1_tool", "a1_y_link"]},
        {"name": "a2", "group": "arm_2", "joints": ["a2_x", "a2_y"], "start": [0.8, 0.0],
         "limits": [[-2, -2], [2, 2]], "attach_link": "a2_tool"},
    ],
    "collision_objects": [
        {"id": "center_obstacle", "mesh": "package://p/m/center.stl",
         "pose": {"position": [0, 0, 0]}, "movable": False},
        {"id": "box1", "mesh": "package://p/m/box.stl",
         "pose": {"position": [0.8, -0.8, 0]}, "movable": True},
    ],
    "tasks": [
        {"name": "a1_pick", "robots": ["a1"], "type": "pick",
         "goal": {"type": "single", "config": [0.8, -0.8]},
         "attach": {"object": "box1", "link": "a1_tool",
                    "grasp": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}}},
        {"name": "a1_place", "robots": ["a1"], "type": "place",
         "goal": {"type": "single", "config": [-0.8, -0.8]},
         "detach": {"object": "box1", "place": {"position": [-0.8, -0.8, 0]}}},
        {"name": "a2_goal", "robots": ["a2"], "goal": {"type": "single", "config": [0.0, 0.8]}},
        {"name": "terminal", "robots": ["a1", "a2"],
         "goal": {"type": "single", "config": [-0.8, -0.8, 0.0, 0.8]}},
    ],
    "dependencies": [["a1_place", "a1_pick"], ["terminal", "a1_place"], ["terminal", "a2_goal"]],
}


# --------------------------------------------------- manipulation: spec
def test_spec_parses_manipulation_and_movable():
    spec = MoveItProblemSpec.from_dict(PICK_PLACE)
    assert spec.has_manipulation
    assert [o.id for o in spec.movable_objects] == ["box1"]
    assert [o.id for o in spec.static_objects] == ["center_obstacle"]

    pick = next(t for t in spec.tasks if t.name == "a1_pick")
    place = next(t for t in spec.tasks if t.name == "a1_place")
    assert pick.manipulation.kind == "pick" and pick.manipulation.obj == "box1"
    assert pick.manipulation.link == "a1_tool"
    assert pick.manipulation.pose[:3] == (0.0, 0.0, 0.0)
    assert place.manipulation.kind == "place"
    assert place.manipulation.pose[:3] == (-0.8, -0.8, 0.0)
    # a plain goal task has no manipulation
    assert next(t for t in spec.tasks if t.name == "a2_goal").manipulation is None


def test_spec_rejects_attach_of_non_movable():
    bad = json.loads(json.dumps(PICK_PLACE))
    bad["tasks"][0]["attach"]["object"] = "center_obstacle"  # not movable
    with pytest.raises(ValueError):
        MoveItProblemSpec.from_dict(bad)


# --------------------------------------------------- manipulation: env / sg
def test_env_scene_graph_attaches_and_detaches():
    env = get_env_by_name("moveit.pick_place_dependency")
    assert env.manipulating_env
    assert env.movable_objects == {"box1"}
    # start: box1 rests at the world anchor
    link, pose = env.start_mode.sg["box1"]
    assert link == env.anchor_link
    assert tuple(pose[:3]) == (0.8, -0.8, 0.0)

    # walk the planned mode chain and check attach -> detach
    path, _ = _plan(env, seed=0, t=10.0)
    assert path is not None and env.is_valid_plan(path)
    states = {}
    for s in path:
        states[tuple(s.mode.task_ids)] = s.mode.sg["box1"]
    # while a1 carries it (doing a1_place), box1 is attached to the gripper
    carry = states[(1, 3)]
    assert carry[0] == "a1_tool" and bool_held(env, carry)
    # at the end it has been placed back into the world
    placed = states[(3, 3)]
    assert placed[0] == env.anchor_link and tuple(placed[1][:3]) == (-0.8, -0.8, 0.0)


def bool_held(env, sg_value):
    return sg_value[0] != env.anchor_link


def test_attachments_for_mode_descriptors():
    env = get_env_by_name("moveit.pick_place_dependency")
    # find a held mode by advancing the scene graph manually via the planner
    path, _ = _plan(env, seed=0, t=10.0)
    held_state = next(s for s in path if s.mode.sg["box1"][0] == "a1_tool")
    atts = env.attachments_for_mode(held_state.mode)
    assert len(atts) == 1
    a = atts[0]
    assert a.object_id == "box1" and a.link == "a1_tool" and a.held is True
    assert a.touch_links  # carrying arm links so it doesn't self-collide
    assert a.mesh.endswith("box.stl")


def test_pick_place_trajectory_embeds_attachments():
    env = get_env_by_name("moveit.pick_place_dependency")
    path, _ = _plan(env, seed=0, t=10.0)
    traj = env.to_display_trajectory(path, split_by_mode=True)
    segs = traj["metadata"]["segments"]
    # at least one segment has box1 held by the gripper
    held = [
        a
        for seg in segs
        for a in seg["attached_collision_objects"]
        if a["object_id"] == "box1" and a["held"]
    ]
    assert held, "expected box1 to be attached to the gripper in some segment"
    assert held[0]["link"] == "a1_tool"
    # every segment lists box1 (resting or held)
    for seg in segs:
        ids = {a["object_id"] for a in seg["attached_collision_objects"]}
        assert "box1" in ids


def test_build_display_trajectory_msg_requires_ros():
    # Without moveit_msgs available, building the real ROS message raises a clear
    # ImportError (the dict form above is what tests rely on).
    from multi_robot_multi_goal_planning.ros.trajectory_conversion import (
        build_display_trajectory_msg,
    )
    spec = MoveItProblemSpec.from_dict(SPEC)
    env = build_moveit_env(spec, _analytic_checker())
    path, _ = _plan(env)
    traj = to_display_trajectory_dict(path, env.joint_names)
    try:
        import moveit_msgs  # noqa: F401
        has_ros = True
    except ImportError:
        has_ros = False
    if not has_ros:
        with pytest.raises(ImportError):
            build_display_trajectory_msg(traj)
