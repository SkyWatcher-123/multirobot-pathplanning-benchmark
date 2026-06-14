"""Declarative problem specification for MoveIt-backed planning problems.

A problem is described by a plain dict (loadable from JSON or YAML) that captures
exactly the assumptions for the ROS1 / MoveIt deployment:

* robots are ROS1 robots described by URDF/SRDF -- here we only need each robot's
  MoveIt planning *group*, its ordered list of actuated *joints*, a *start*
  configuration and joint *limits* (limits may be omitted and read from the URDF
  on the parameter server at runtime);
* every planning-scene object is a *mesh* (``package://`` / file path) added as a
  MoveIt collision object;
* the task ordering is given by a *dependency graph* (or, optionally, a fully
  ordered sequence).

This module performs no ROS work -- it just parses the spec into the planning
primitives (:class:`Task`, :class:`DependencyGraph`, configurations, limits and a
list of mesh descriptors). :mod:`moveit_env` turns those into a runnable
:class:`BaseProblem`. Keeping it ROS-free makes it unit-testable without a ROS
install.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .configuration import NpConfiguration
from .dependency_graph import DependencyGraph
from .goals import Goal, SingleGoal, GoalRegion
from .planning_env import Task


@dataclass
class MeshObject:
    """A mesh collision object to add to the MoveIt planning scene."""

    id: str
    mesh: str  # package:// URI or absolute path to an STL/DAE/PLY mesh
    frame: str = "world"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # x,y,z,w
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MeshObject":
        pose = d.get("pose", {}) or {}
        pos = pose.get("position", d.get("position", [0.0, 0.0, 0.0]))
        ori = pose.get("orientation", d.get("orientation", [0.0, 0.0, 0.0, 1.0]))
        scale = d.get("scale", [1.0, 1.0, 1.0])
        return cls(
            id=d["id"],
            mesh=d["mesh"],
            frame=d.get("frame", "world"),
            position=tuple(float(x) for x in pos),
            orientation=tuple(float(x) for x in ori),
            scale=tuple(float(x) for x in scale),
        )


@dataclass
class RobotSpec:
    """A single robot: its MoveIt group, ordered joints, start config and limits."""

    name: str
    group: str
    joints: List[str]
    start: List[float]
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None
    # Optional link a grasped object is attached to (used for manipulation modes).
    attach_link: Optional[str] = None

    @property
    def dim(self) -> int:
        return len(self.joints)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RobotSpec":
        joints = list(d["joints"])
        start = list(d.get("start", [0.0] * len(joints)))
        if len(start) != len(joints):
            raise ValueError(
                f"robot '{d.get('name')}' start has {len(start)} values but "
                f"{len(joints)} joints"
            )
        limits = d.get("limits")
        lower = upper = None
        if limits is not None:
            lower = [float(x) for x in limits[0]]
            upper = [float(x) for x in limits[1]]
        return cls(
            name=d["name"],
            group=d.get("group", d["name"]),
            joints=joints,
            start=[float(x) for x in start],
            lower=lower,
            upper=upper,
            attach_link=d.get("attach_link"),
        )


def _build_goal(robots: List[str], robot_dims: Dict[str, int], goal_spec: Any) -> Goal:
    """Build a :class:`Goal` from the spec for the involved robots.

    A goal is defined in the joint space of the *involved* robots concatenated in
    the order they appear in ``robots`` (consistent with how the benchmark
    evaluates ``goal.satisfies_constraints`` on ``q_concat``).
    """
    expected_dim = sum(robot_dims[r] for r in robots)

    # Bare list -> a single joint-space goal.
    if isinstance(goal_spec, (list, tuple)):
        cfg = np.array([float(x) for x in goal_spec], dtype=float)
        if cfg.shape[0] != expected_dim:
            raise ValueError(
                f"goal config has dim {cfg.shape[0]} but involved robots need "
                f"{expected_dim}"
            )
        return SingleGoal(cfg)

    if not isinstance(goal_spec, dict):
        raise ValueError(f"unsupported goal spec: {goal_spec!r}")

    gtype = goal_spec.get("type", "single")
    if gtype in ("single", "config", "joint"):
        cfg = np.array([float(x) for x in goal_spec["config"]], dtype=float)
        if cfg.shape[0] != expected_dim:
            raise ValueError(
                f"goal config has dim {cfg.shape[0]} but involved robots need "
                f"{expected_dim}"
            )
        return SingleGoal(cfg)
    if gtype in ("region", "box"):
        lower = np.array([float(x) for x in goal_spec["lower"]], dtype=float)
        upper = np.array([float(x) for x in goal_spec["upper"]], dtype=float)
        return GoalRegion(np.vstack([lower, upper]))

    raise ValueError(f"unknown goal type '{gtype}'")


@dataclass
class MoveItProblemSpec:
    """Parsed problem specification ready to be consumed by ``moveit_env``."""

    robots: List[RobotSpec]
    tasks: List[Task]
    base_frame: str = "world"
    mode_logic: str = "dependency"  # "dependency" or "sequence"
    dependencies: List[Tuple[str, str]] = field(default_factory=list)
    sequence: List[str] = field(default_factory=list)
    collision_objects: List[MeshObject] = field(default_factory=list)
    collision_tolerance: float = 0.0
    collision_resolution: float = 0.05
    velocity: float = 0.25
    name: str = "moveit_problem"

    # ------------------------------------------------------------------ helpers
    @property
    def robot_names(self) -> List[str]:
        return [r.name for r in self.robots]

    @property
    def robot_dims(self) -> Dict[str, int]:
        return {r.name: r.dim for r in self.robots}

    @property
    def joint_names(self) -> List[str]:
        """All actuated joints, concatenated in robot order (== config order)."""
        names: List[str] = []
        for r in self.robots:
            names.extend(r.joints)
        return names

    def start_configuration(self) -> NpConfiguration:
        return NpConfiguration.from_list([np.array(r.start, dtype=float) for r in self.robots])

    def limits(self) -> Optional[np.ndarray]:
        """Stacked ``[lower; upper]`` joint limits, or ``None`` if any are missing.

        When ``None``, ``moveit_env`` reads the limits from the URDF on the
        parameter server instead.
        """
        lowers: List[float] = []
        uppers: List[float] = []
        for r in self.robots:
            if r.lower is None or r.upper is None:
                return None
            lowers.extend(r.lower)
            uppers.extend(r.upper)
        return np.vstack([np.array(lowers, dtype=float), np.array(uppers, dtype=float)])

    def build_dependency_graph(self) -> DependencyGraph:
        graph = DependencyGraph()
        task_names = {t.name for t in self.tasks}
        # Make sure every task is a node even if it has no edges.
        for name in task_names:
            graph.add_node(name)
        for dependent, dependency in self.dependencies:
            if dependent not in task_names:
                raise ValueError(f"dependency refers to unknown task '{dependent}'")
            if dependency not in task_names:
                raise ValueError(f"dependency refers to unknown task '{dependency}'")
            graph.add_dependency(dependent, dependency)
        return graph

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoveItProblemSpec":
        robots = [RobotSpec.from_dict(r) for r in d["robots"]]
        robot_dims = {r.name: r.dim for r in robots}

        tasks: List[Task] = []
        for t in d["tasks"]:
            involved = list(t["robots"])
            goal = _build_goal(involved, robot_dims, t["goal"])
            tasks.append(
                Task(
                    name=t["name"],
                    robots=involved,
                    goal=goal,
                    type=t.get("type"),
                    frames=t.get("frames"),
                    side_effect=t.get("side_effect"),
                )
            )

        collision_objects = [MeshObject.from_dict(o) for o in d.get("collision_objects", [])]

        col = d.get("collision", {}) or {}
        dependencies = [tuple(e) for e in d.get("dependencies", [])]

        return cls(
            robots=robots,
            tasks=tasks,
            base_frame=d.get("base_frame", "world"),
            mode_logic=d.get("mode_logic", "dependency"),
            dependencies=dependencies,
            sequence=list(d.get("sequence", [])),
            collision_objects=collision_objects,
            collision_tolerance=float(col.get("tolerance", 0.0)),
            collision_resolution=float(col.get("resolution", 0.05)),
            velocity=float(d.get("velocity", 0.25)),
            name=d.get("name", "moveit_problem"),
        )

    @classmethod
    def from_json(cls, text: str) -> "MoveItProblemSpec":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, text: str) -> "MoveItProblemSpec":
        try:
            import yaml  # type: ignore
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to load a YAML problem spec; install pyyaml "
                "or use a JSON spec instead."
            ) from err
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_file(cls, filename: str) -> "MoveItProblemSpec":
        with open(filename, "r") as f:
            text = f.read()
        if filename.endswith((".yaml", ".yml")):
            return cls.from_yaml(text)
        if filename.endswith(".json"):
            return cls.from_json(text)
        # Fall back on content sniffing: try JSON first, then YAML.
        try:
            return cls.from_json(text)
        except json.JSONDecodeError:
            return cls.from_yaml(text)
