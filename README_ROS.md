# `mrmg_moveit_planning` — ROS1 Noetic + MoveIt package

This repository doubles as a **ROS1 Noetic catkin package** that runs the
multi-robot–multi-goal benchmark planners (RRT\*, BiRRT\*, PRM, AIT\*, EIT\*)
against a **MoveIt planning scene**, and replays the resulting multi-modal plan in
**MoveIt RViz**.

It is built around the benchmark's existing backend-agnostic design: every
planner only depends on the `BaseProblem` interface, and there are already
URDF/mesh backends (pinocchio, mujoco). The MoveIt integration is just another
backend plus a thin ROS layer — **no change to the planners**, and the base
planner does **not** use MoveIt's OMPL pipeline.

```
   problem spec (dependency graph + robots + mesh objects)         [you provide]
            │  MoveItProblemSpec.from_file(...)
            ▼
   MoveItEnvironment  ── implements ──►  BaseProblem
            │   collision check delegated to ▼
            │                 /check_state_validity  (move_group, meshes in scene)
            ▼
   benchmark planner (RRT* by default)  ──►  List[State]  (multi-modal path)
            │   ros/trajectory_conversion.py
            ▼
   moveit_msgs/DisplayTrajectory  ──►  /move_group/display_planned_path  ──►  RViz replay
```

---

## What you get

| Concern (your assumptions) | How it is handled |
|---|---|
| Task ordering via a **dependency graph** | `mode_logic: dependency` in the problem spec → `DependencyGraphMixin`. A fully-ordered `sequence` is also supported. |
| Robots are **ROS1 robots with URDF/SRDF** | Robots are referenced by their MoveIt **group** + ordered **joints**; kinematics/limits come from `robot_description` on the param server. |
| Planning-scene objects are **meshes** (MoveIt collision) | `collision_objects` are meshes added to the MoveIt planning scene via `PlanningSceneInterface.add_mesh`. |
| **Grasping**: a task's goal specifies which mesh attaches to the end-effector (or none) | `movable` meshes are carried as `AttachedCollisionObject`s during planning: a `pick` task attaches the mesh to the gripper link with the given grasp pose; a `place` task returns it to the world. Collision checks (and RViz) follow the carried mesh. |
| **Replayable in MoveIt RViz** | The path is published as `moveit_msgs/DisplayTrajectory` on `/move_group/display_planned_path` and exported to JSON for offline replay. |
| Base planner need not be **OMPL** | Planning is done by the benchmark's own samplers (RRT\* default); move_group is used only for collision checking + scene + RViz. |

---

## Package layout

```
package.xml, CMakeLists.txt, setup.py     catkin manifest / build / python export
scripts/mrmg_plan_node                     rosrun entry: plan + replay
scripts/mrmg_replay_node                   rosrun entry: replay an exported trajectory
launch/                                    demo.launch, move_group.launch, plan.launch,
                                           replay.launch, planning_context.launch,
                                           moveit_rviz.launch, template_robot.launch
config/                                    demo MoveIt config (SRDF, kinematics, limits, OMPL)
config/problems/                           example problem specs (demo_dependency.json, template.json)
rviz/moveit.rviz                           RViz config (MotionPlanning display, looped)
urdf/, meshes/                             demo two-arm robot + obstacle mesh
src/multi_robot_multi_goal_planning/
  problems/moveit_env.py                   MoveItEnvironment (BaseProblem) + checker strategy
  problems/moveit_problem_spec.py          spec parser (robots/tasks/meshes/dependency graph)
  ros/trajectory_conversion.py             path -> DisplayTrajectory (ROS-independent)
  ros/moveit_interface.py                  planning scene + /check_state_validity + publishing
  ros/plan_node.py, ros/replay_node.py     node implementations
  ros/mesh_resolver.py                     package:// mesh URI resolution
```

---

## Build

Place (or symlink) this repository into a catkin workspace and build it:

```bash
cd ~/catkin_ws/src
ln -s /path/to/multirobot-pathplanning-benchmark mrmg_moveit_planning
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y     # MoveIt, rviz, etc.
catkin build         # or: catkin_make
source devel/setup.bash
```

The Python planning core (`multi_robot_multi_goal_planning`) is exposed to ROS via
`catkin_python_setup()` (see `setup.py`). Its numeric dependencies
(`numpy`, `scipy`, `networkx`, `numba`, `sortedcontainers`) come from
`pyproject.toml`; install them into the Python environment ROS uses, e.g.
`python3 -m pip install numpy scipy networkx numba sortedcontainers pyyaml`.

> Noetic ships Python 3.8. `numba` is only needed by some distance helpers; the
> MoveIt backend and RRT\* run fine with it installed via pip.

---

## Quick start (bundled demo)

The demo is two planar arms that must swap sides around a central box obstacle,
with a dependency graph over the tasks. It is self-contained so the whole pipeline
runs without any extra robot description.

Terminal 1 — bring up move_group + RViz:

```bash
roslaunch mrmg_moveit_planning demo.launch
```

Terminal 2 — plan with RRT\* and replay in RViz:

```bash
roslaunch mrmg_moveit_planning plan.launch          # default: RRT* on the demo dependency problem
# or choose a planner / problem:
roslaunch mrmg_moveit_planning plan.launch planner:=birrt_star max_time:=15
```

You should see the two arms animate around the obstacle in RViz (the
MotionPlanning display loops `/move_group/display_planned_path`). The trajectory
is also written to `~/mrmg_last_trajectory.json`.

Replay the saved plan later (no re-planning):

```bash
roslaunch mrmg_moveit_planning replay.launch file:=$HOME/mrmg_last_trajectory.json
```

### Pick-and-place demo (attach / detach)

A second bundled problem has robot `a1` pick up a movable box, carry it around the
obstacle, and place it — under a dependency graph (`a1_place` depends on
`a1_pick`). The grasped mesh is attached to the gripper for collision checking and
follows it in RViz:

```bash
roslaunch mrmg_moveit_planning plan.launch \
    problem_spec:=$(rospack find mrmg_moveit_planning)/config/problems/pick_place_dependency.json
```

---

## Using your own robot(s)

1. **MoveIt config.** Generate one for your robot(s) with the MoveIt Setup
   Assistant (URDF/SRDF, `kinematics.yaml`, `joint_limits.yaml`, `move_group.launch`,
   `moveit.rviz`). For multiple robots, build **one combined URDF** containing all
   of them and define a group per robot (plus optionally a combined group).

2. **Problem spec.** Copy `config/problems/template.json` and fill in:
   - `robots`: each robot's MoveIt `group`, ordered `joints`, and `start` config
     (omit `limits` to read them from the URDF at runtime);
   - `collision_objects`: your mesh obstacles (`package://…` or file paths);
   - `tasks`: joint-space goals (a goal's `config` spans the task's `robots`,
     concatenated in order);
   - `dependencies`: the dependency graph as `[A, B]` edges ("A depends on B"),
     with exactly one leaf task (conventionally `terminal`).

3. **Run.** Launch your robot's `move_group` + RViz, then:

   ```bash
   roslaunch mrmg_moveit_planning template_robot.launch \
       problem_spec:=/path/to/your_problem.json planner:=rrt_star
   ```

If your MoveIt build rejects an empty group in `/check_state_validity`, pass
`check_group:=<a group spanning all robots>`.

### Problem spec reference

```jsonc
{
  "base_frame": "world",
  "mode_logic": "dependency",          // or "sequence" (+ a "sequence": [...] list)
  "velocity": 0.25,                     // joint-space speed for trajectory timing
  "collision": {"tolerance": 0.0, "resolution": 0.05},
  "robots": [
    {"name": "a1", "group": "arm_1", "joints": ["j1","j2"], "start": [0,0],
     "limits": [[lo,...],[hi,...]]}    // limits optional
  ],
  "collision_objects": [
    {"id": "table", "mesh": "package://pkg/meshes/table.stl", "frame": "world",
     "pose": {"position": [x,y,z], "orientation": [x,y,z,w]}, "scale": [1,1,1]}
  ],
  "tasks": [
    {"name": "a1_goal", "robots": ["a1"], "goal": {"type": "single", "config": [...]}},
    {"name": "terminal", "robots": ["a1","a2"], "goal": {"type": "single", "config": [...]}}
  ],
  "dependencies": [["terminal", "a1_goal"]]
}
```

Goal types: `single` (a joint config) and `region`/`box` (`lower`/`upper` bounds).
Specs may be JSON or YAML.

#### Grasping (attach / detach)

Mark a mesh `"movable": true` and add an `attach`/`detach` block to the task whose
goal pose performs the grasp/release. The task's goal is the pose at which the
side effect happens; the block says *which* mesh and *how* it attaches:

```jsonc
// in collision_objects:
{"id": "box1", "mesh": "package://pkg/meshes/box.stl",
 "pose": {"position": [0.5, -0.4, 0.1]}, "movable": true}

// pick task: when robot_a reaches this goal, box1 attaches to its end-effector
{"name": "pick_box1", "robots": ["robot_a"], "type": "pick",
 "goal": {"type": "single", "config": [...]},
 "attach": {"object": "box1", "link": "robot_a_tool",
            "grasp": {"position": [0,0,0.02], "orientation": [0,0,0,1]}}}

// place task: when reached, box1 is released into the world at this pose
{"name": "place_box1", "robots": ["robot_a"], "type": "place",
 "goal": {"type": "single", "config": [...]},
 "detach": {"object": "box1", "place": {"position": [-0.5, -0.4, 0.1]}}}
```

`link` defaults to the robot's `attach_link`; set `touch_links` on the robot to
the gripper links so a held object does not self-collide with the carrying arm.
Movable objects are *not* added as world geometry — they ride along as attached
objects (anchored to `anchor_link` while resting), so they are never
double-counted. A task with no `attach`/`detach` is a plain "goto".

---

## How it works

- **Collision checking** (`ros/moveit_interface.py`): `MoveItServiceCollisionChecker`
  fills a `moveit_msgs/RobotState` with the full joint vector and calls
  `/check_state_validity` (`moveit_msgs/GetStateValidity`). The mesh
  `collision_objects` are added to the planning scene first, so the check accounts
  for robot↔scene and robot↔robot collisions. Edge checks discretize the
  straight-line motion (binary-search ordering, same as the other backends).
- **Planning**: any benchmark planner runs unchanged on `MoveItEnvironment`
  (`_build_planner` in `ros/plan_node.py` maps `planner:=` to the class). RRT\* is
  the default.
- **Replay** (`ros/trajectory_conversion.py`): the `List[State]` is split into
  contiguous mode segments, each emitted as a `RobotTrajectory` with a constant
  max-joint-speed time parameterization, wrapped in a `DisplayTrajectory`, and
  published latched on `/move_group/display_planned_path`.

The conversion and spec modules never import `rospy`, so they are unit-testable
without ROS (see `tests/test_moveit_backend.py`) and the path can be exported to
JSON and replayed offline.

---

## Tests

ROS-independent tests (run anywhere):

```bash
pip install numpy numba networkx scipy sortedcontainers pyyaml pytest matplotlib
PYTHONPATH=src pytest tests/test_moveit_backend.py
```

These cover the spec parser, the MoveIt environment with an injected analytic
checker, planning with RRT\*, and the `DisplayTrajectory` conversion/JSON
round-trip.

The original benchmark tests in `tests/test_planners.py` are kept as examples;
the `rai`/`pinocchio` cases require those optional backends. (The
abstract-environment example runs without any backend.)

---

## Notes & limitations

- **Manipulation / attached objects.** Both the static path and pick/place
  manipulation are supported. During planning, a movable mesh is represented as an
  `AttachedCollisionObject` whose parent link changes per mode (world anchor when
  resting, gripper link when held), built from the per-mode scene graph
  (`MoveItEnvironment.get_scenegraph_info_for_mode` / `attachments_for_mode`) and
  applied to each `/check_state_validity` request — so the carried mesh is checked
  against the world and the other robots, and is removed from the world while
  held. In RViz the plan is replayed segment-by-segment so the grasped mesh
  attaches and detaches. See `moveit.pick_place_dependency` /
  `config/problems/pick_place_dependency.json`. The grasp/place poses are taken
  from the spec (no IK needed in the env).
- **No live ROS in CI.** This package was developed where ROS is unavailable, so
  the ROS-touching code (`moveit_interface`, the nodes, launch/config) is
  validated by construction and by the import-guards; run `catkin build` + the
  demo on a Noetic + MoveIt machine for end-to-end validation.
