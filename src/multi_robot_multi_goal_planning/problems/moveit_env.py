"""A MoveIt-backed planning problem for ROS1 Noetic.

This is a backend for the benchmark in the same spirit as ``pinocchio_env`` and
``mujoco_env``: it implements :class:`BaseProblem` so every planner in this
repository (RRT*, BiRRT*, PRM, AIT*, EIT*, prioritized, receding-horizon) can run
on it unchanged -- there is no dependency on MoveIt's OMPL pipeline.

Collision checking is delegated to MoveIt through a *pluggable* state-validity
checker:

* In a live ROS deployment the checker calls ``move_group``'s
  ``/check_state_validity`` (``moveit_msgs/GetStateValidity``) service against the
  MoveIt planning scene (which holds the mesh collision objects). That checker
  lives in :mod:`multi_robot_multi_goal_planning.ros.moveit_interface` and is
  injected at runtime, so importing this module never requires ROS.
* For headless testing / development you can inject any
  ``callable(joint_names, positions, mode) -> bool`` (see
  :class:`CallableStateValidityChecker`).

The module is import-safe without ROS: it only depends on numpy and the planning
core. The ROS plumbing (planning scene, validity service, RViz publishing) is in
the :mod:`multi_robot_multi_goal_planning.ros` subpackage.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .configuration import (
    Configuration,
    NpConfiguration,
    config_cost,
    config_dist,
    batch_config_cost,
)
from .dependency_graph import DependencyGraph
from .planning_env import (
    AgentType,
    BaseModeLogic,
    BaseProblem,
    ConstraintType,
    DependencyGraphMixin,
    DependencyType,
    DynamicsType,
    GoalType,
    ManipulationType,
    Mode,
    ProblemSpec,
    SafePoseType,
    SequenceMixin,
    State,
    Task,
    generate_binary_search_indices,
)
from .moveit_problem_spec import MoveItProblemSpec
from .registry import register

from dataclasses import dataclass


@dataclass
class Attachment:
    """A movable mesh attached to a link in a given mode.

    ``pose`` is the object's ``(x,y,z,qx,qy,qz,qw)`` transform in ``link``'s frame.
    ``held`` is True when attached to a gripper link (grasped) rather than the
    world anchor link (resting).
    """

    object_id: str
    link: str
    pose: tuple
    mesh: str
    scale: tuple
    touch_links: list
    held: bool


# --------------------------------------------------------------------------- #
#  Collision checking strategy
# --------------------------------------------------------------------------- #
class StateValidityChecker(ABC):
    """Strategy object that decides whether a full-scene state is collision free.

    Implementations receive the flat joint vector (ordered like ``joint_names``)
    and the current :class:`Mode` (so manipulation backends can apply the right
    attached objects), and return ``True`` when the state is valid.
    """

    @abstractmethod
    def is_valid(
        self, joint_names: List[str], positions: NDArray, mode: Optional[Mode]
    ) -> bool:  # pragma: no cover - interface
        ...


class CallableStateValidityChecker(StateValidityChecker):
    """Adapt a plain callable into a :class:`StateValidityChecker`.

    Handy for headless tests (e.g. an analytic obstacle) and for wiring a custom
    checker without subclassing.
    """

    def __init__(self, fn: Callable[[List[str], NDArray, Optional[Mode]], bool]):
        self._fn = fn

    def is_valid(self, joint_names, positions, mode) -> bool:
        return bool(self._fn(joint_names, positions, mode))


class _UnsetStateValidityChecker(StateValidityChecker):
    """Default checker that fails loudly until a real one is injected."""

    def is_valid(self, joint_names, positions, mode) -> bool:  # pragma: no cover
        raise RuntimeError(
            "No state-validity checker set on this MoveIt environment. Call "
            "env.set_validity_checker(...) with a MoveItServiceCollisionChecker "
            "(from multi_robot_multi_goal_planning.ros.moveit_interface) in a live "
            "ROS session, or inject a CallableStateValidityChecker for headless "
            "testing."
        )


# --------------------------------------------------------------------------- #
#  Base MoveIt environment (backend methods, mixin-agnostic)
# --------------------------------------------------------------------------- #
class MoveItEnvironment(BaseProblem):
    """Backend half of a MoveIt planning problem.

    Concrete problems combine this with a mode-logic mixin
    (:class:`SequenceMixin` or :class:`DependencyGraphMixin`); see
    :func:`build_moveit_env`.
    """

    def _init_backend(
        self,
        spec: MoveItProblemSpec,
        validity_checker: Optional[StateValidityChecker] = None,
    ) -> None:
        self.problem_spec = spec
        self.base_frame = spec.base_frame
        self.velocity = spec.velocity

        self.robots = spec.robot_names
        self.robot_dims = spec.robot_dims
        self.robot_joint_names = {r.name: list(r.joints) for r in spec.robots}
        self.robot_groups = {r.name: r.group for r in spec.robots}
        self.attach_links = {r.name: r.attach_link for r in spec.robots}
        self.joint_names = spec.joint_names
        self.collision_objects = spec.collision_objects
        self.anchor_link = spec.anchor_link

        # Manipulation bookkeeping: movable objects ride along as attached
        # collision objects whose parent link changes per mode.
        self.movable_objects = {o.id for o in spec.movable_objects}
        self._movable_specs = {o.id: o for o in spec.movable_objects}
        # link -> links a grasped object may touch (the carrying gripper links).
        self._touch_links_by_link: Dict[str, List[str]] = {}
        for r in spec.robots:
            if r.attach_link is not None:
                self._touch_links_by_link[r.attach_link] = (
                    list(r.touch_links) if r.touch_links else [r.attach_link]
                )

        # Build robot_idx (slice into the flat configuration) in robot order.
        offset = 0
        self.robot_idx: Dict[str, List[int]] = {}
        for r in spec.robots:
            self.robot_idx[r.name] = [offset + i for i in range(r.dim)]
            offset += r.dim

        self.start_pos = spec.start_configuration()

        limits = spec.limits()
        if limits is None:
            # Fall back to wide default limits; the ROS layer can overwrite these
            # with the real URDF limits via set_limits_from_dict(...).
            n = self.start_pos.state().shape[0]
            limits = np.vstack([-np.ones(n) * np.pi, np.ones(n) * np.pi])
        self.limits = limits

        self.collision_tolerance = spec.collision_tolerance
        self.collision_resolution = spec.collision_resolution

        self.cost_metric = "euclidean"
        self.cost_reduction = "max"

        self.manipulating_env = spec.has_manipulation
        self._current_mode: Optional[Mode] = None
        self._display_fn: Optional[Callable[[List[State]], None]] = None

        # Initial scene graph: every movable object rests at its world pose,
        # anchored to the fixed root link.
        self.initial_sg: Dict[str, tuple] = {
            o.id: (self.anchor_link, tuple(o.pose7)) for o in spec.movable_objects
        }

        self._validity_checker: StateValidityChecker = (
            validity_checker or _UnsetStateValidityChecker()
        )

        n_robots = len(self.robots)
        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT if n_robots > 1 else AgentType.SINGLE_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=(
                ManipulationType.MANIPULATION
                if spec.has_manipulation
                else ManipulationType.STATIC
            ),
            dependency=(
                DependencyType.UNORDERED
                if spec.mode_logic == "dependency"
                else DependencyType.FULLY_ORDERED
            ),
            dynamics=DynamicsType.GEOMETRIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

    # ------------------------------------------------------------------ config
    def set_validity_checker(self, checker) -> None:
        """Inject the collision checker (e.g. the MoveIt ``/check_state_validity``)."""
        if not isinstance(checker, StateValidityChecker):
            checker = CallableStateValidityChecker(checker)
        self._validity_checker = checker

    def set_display_function(self, fn: Callable[[List[State]], None]) -> None:
        """Inject the callback used by :meth:`display_path` (publishes to RViz)."""
        self._display_fn = fn

    def set_limits_from_dict(self, joint_limits: Dict[str, "tuple"]) -> None:
        """Set joint limits from a ``{joint_name: (lower, upper)}`` mapping.

        Used by the ROS layer to populate sampling limits from the URDF on the
        parameter server when the spec does not provide them.
        """
        lowers, uppers = [], []
        for name in self.joint_names:
            lo, hi = joint_limits[name]
            lowers.append(float(lo))
            uppers.append(float(hi))
        self.limits = np.vstack([np.array(lowers), np.array(uppers)])

    # ------------------------------------------------------------- collision
    def is_collision_free(
        self, q: Optional[Configuration], mode: Optional[Mode]
    ) -> bool:
        if q is None:
            raise ValueError("is_collision_free requires a configuration")
        positions = q.state() if isinstance(q, Configuration) else np.asarray(q)
        return self._validity_checker.is_valid(self.joint_names, positions, mode)

    def is_collision_free_for_robot(
        self,
        r,
        q: NDArray,
        m: Optional[Mode] = None,
        collision_tolerance: Optional[float] = None,
        set_mode: bool = True,
    ) -> bool:
        # Whole-scene validity check; the per-robot variant simply reuses it.
        return self.is_collision_free(NpConfiguration.from_numpy(np.asarray(q)), m)

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        mode: Mode,
        resolution: Optional[float] = None,
        tolerance: Optional[float] = None,
        include_endpoints: bool = False,
        N_start: int = 0,
        N_max: Optional[int] = None,
        N: Optional[int] = None,
    ) -> bool:
        """Collision-check the straight-line edge between ``q1`` and ``q2``.

        Identical discretisation strategy to the other backends (binary-search
        ordering so collisions near the middle are found fast).
        """
        if resolution is None:
            resolution = self.collision_resolution

        if N is None:
            N = int(config_dist(q1, q2, "max") / resolution) + 1
            N = max(2, N)

        if N_start > N:
            assert False

        if N_max is None:
            N_max = N
        N_max = min(N, N_max)

        idx = generate_binary_search_indices(N)

        q1_state = q1.state()
        q2_state = q2.state()
        direction = (q2_state - q1_state) / (N - 1)

        for i in idx[N_start:N_max]:
            if not include_endpoints and (i == 0 or i == N - 1):
                continue
            q = q1_state + direction * i
            if not self.is_collision_free(NpConfiguration(q, q1._array_slice), mode):
                return False

        return True

    # ------------------------------------------------------------------- modes
    def get_scenegraph_info_for_mode(self, mode: Mode, is_start_mode: bool = False):
        """Scene graph for a mode: which movable object is parented to which link.

        For a static scene this is empty. For a manipulation scene each movable
        object maps to ``(link_name, pose7)`` where ``pose7`` is the object's
        ``(x,y,z,qx,qy,qz,qw)`` transform in ``link_name``'s frame:

        * resting in the world -> anchored to ``self.anchor_link`` at its world pose;
        * grasped -> attached to the gripper link with the task's grasp pose.

        The graph is derived incrementally from the previous mode plus the
        attach/detach side effect of the task that was just completed to enter this
        mode (mirrors the pinocchio backend, but with explicit poses so no FK is
        needed here -- the goal pose specifies exactly which mesh attaches and how).
        """
        if not self.manipulating_env:
            return {}

        prev = mode.prev_mode
        if prev is None or is_start_mode:
            return dict(self.initial_sg)

        sg = dict(prev.sg)
        completed_task = self.get_active_task(prev, mode.task_ids)
        manip = getattr(completed_task, "manipulation", None)
        if manip is not None:
            if manip.kind == "pick":
                sg[manip.obj] = (manip.link, tuple(manip.pose))
            elif manip.kind == "place":
                sg[manip.obj] = (self.anchor_link, tuple(manip.pose))
        return sg

    def attachments_for_mode(self, mode: Optional[Mode]) -> List["Attachment"]:
        """Resolve a mode's scene graph into concrete attachment descriptors.

        Each entry pairs a movable mesh (geometry/scale from the spec) with the
        link it is attached to and its relative pose in that mode. Consumed by the
        ROS checker (to build ``AttachedCollisionObject`` messages) and by the
        trajectory export (so the carried mesh shows up in RViz).
        """
        if not self.manipulating_env or mode is None or not mode.sg:
            return []
        out: List[Attachment] = []
        for obj_id, value in mode.sg.items():
            link, pose = value[0], value[1]
            spec = self._movable_specs.get(obj_id)
            if spec is None:
                continue
            held = link != self.anchor_link
            out.append(
                Attachment(
                    object_id=obj_id,
                    link=link,
                    pose=tuple(float(x) for x in pose),
                    mesh=spec.mesh,
                    scale=tuple(spec.scale),
                    touch_links=list(self._touch_links_by_link.get(link, [])),
                    held=held,
                )
            )
        return out

    def set_to_mode(self, mode: Mode) -> None:
        # Remember the mode so the checker (and any attached-object handling)
        # can use it; the per-state validity check carries the attachments, so no
        # global planning-scene mutation is needed here.
        self._current_mode = mode

    # ------------------------------------------------------------------ sampling
    def sample_config_uniform_in_limits(self) -> Configuration:
        rnd = np.random.uniform(low=self.limits[0, :], high=self.limits[1, :])
        return self.start_pos.from_flat(rnd)

    # -------------------------------------------------------------------- costs
    def config_cost(self, start: Configuration, end: Configuration) -> float:
        return config_cost(start, end, self.cost_metric, self.cost_reduction)

    def batch_config_cost(
        self,
        starts,
        ends,
        tmp_agent_slice=None,
    ) -> NDArray:
        return batch_config_cost(
            starts,
            ends,
            self.cost_metric,
            self.cost_reduction,
            tmp_agent_slice=tmp_agent_slice,
        )

    # --------------------------------------------------------------- visualize
    def show(self, blocking: bool = True) -> None:
        # Visualisation happens in RViz for the MoveIt backend.
        print(
            "[MoveItEnvironment] start configuration "
            f"{self.start_pos.state().tolist()} (visualise in RViz)."
        )

    def show_config(self, q: Configuration, blocking: bool = True) -> None:
        print(f"[MoveItEnvironment] configuration {np.asarray(q.state()).tolist()}")

    def to_display_trajectory(self, path: List[State], split_by_mode: bool = True):
        """Convert a planned path into a DisplayTrajectory dict (ROS-independent).

        For manipulation problems the per-mode attachments are embedded so the
        grasped mesh follows the gripper in RViz.
        """
        from multi_robot_multi_goal_planning.ros.trajectory_conversion import (
            to_display_trajectory_dict,
        )

        provider = self.attachments_for_mode if self.manipulating_env else None
        return to_display_trajectory_dict(
            path,
            self.joint_names,
            base_frame=self.base_frame,
            velocity=self.velocity,
            split_by_mode=split_by_mode,
            attachments_provider=provider,
        )

    def export_display_trajectory(
        self, path: List[State], filename: str, split_by_mode: bool = True
    ) -> str:
        """Write a planned path to JSON for offline RViz replay; returns the path."""
        from multi_robot_multi_goal_planning.ros.trajectory_conversion import (
            save_trajectory_json,
        )

        traj = self.to_display_trajectory(path, split_by_mode=split_by_mode)
        save_trajectory_json(traj, filename)
        return filename

    def display_path(
        self,
        path: List[State],
        stop: bool = True,
        export: bool = False,
        pause_time: float = 0.01,
        stop_at_end: bool = False,
        adapt_to_max_distance: bool = False,
        stop_at_mode: bool = False,
    ) -> None:
        if self._display_fn is not None:
            self._display_fn(path)
        else:
            print(
                "[MoveItEnvironment] display_path called but no display function is "
                "wired. Use the ROS plan/replay node to publish the trajectory to "
                "RViz, or env.export_display_trajectory(path, file) to save it."
            )


# --------------------------------------------------------------------------- #
#  Concrete mixin combinations + factory
# --------------------------------------------------------------------------- #
class MoveItSequenceEnvironment(SequenceMixin, MoveItEnvironment):
    """MoveIt problem whose tasks follow a fully ordered sequence."""

    def __init__(
        self,
        spec: MoveItProblemSpec,
        validity_checker: Optional[StateValidityChecker] = None,
    ):
        self._init_backend(spec, validity_checker)
        self.tasks = spec.tasks
        if not spec.sequence:
            raise ValueError("sequence mode_logic requires a non-empty 'sequence'")
        self.sequence = self._make_sequence_from_names(spec.sequence)
        BaseModeLogic.__init__(self)


class MoveItDependencyEnvironment(DependencyGraphMixin, MoveItEnvironment):
    """MoveIt problem whose tasks follow a dependency graph (assumption #1)."""

    def __init__(
        self,
        spec: MoveItProblemSpec,
        validity_checker: Optional[StateValidityChecker] = None,
    ):
        self._init_backend(spec, validity_checker)
        self.tasks = spec.tasks
        self.graph = spec.build_dependency_graph()
        BaseModeLogic.__init__(self)


def build_moveit_env(
    spec: MoveItProblemSpec,
    validity_checker: Optional[StateValidityChecker] = None,
) -> MoveItEnvironment:
    """Build the right MoveIt environment for the spec's mode logic."""
    if spec.mode_logic == "sequence":
        return MoveItSequenceEnvironment(spec, validity_checker)
    if spec.mode_logic == "dependency":
        return MoveItDependencyEnvironment(spec, validity_checker)
    raise ValueError(f"unknown mode_logic '{spec.mode_logic}'")


def make_moveit_env_from_file(
    filename: str,
    validity_checker: Optional[StateValidityChecker] = None,
) -> MoveItEnvironment:
    """Load a problem spec from JSON/YAML and build the MoveIt environment."""
    spec = MoveItProblemSpec.from_file(filename)
    return build_moveit_env(spec, validity_checker)


# --------------------------------------------------------------------------- #
#  Bundled demo problem (template + headless-testable)
# --------------------------------------------------------------------------- #
_ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "desc")
_DEMO_SPEC = os.path.join(_ASSET_DIR, "moveit_demo_dependency.json")
_PICK_PLACE_SPEC = os.path.join(_ASSET_DIR, "moveit_pick_place_dependency.json")


def _demo_analytic_checker() -> CallableStateValidityChecker:
    """A tiny analytic stand-in checker so the bundled demo is runnable headless.

    The real MoveIt scene uses meshes via ``/check_state_validity``; this analytic
    checker only exists so the demo problem (and the full planner -> trajectory
    pipeline) can be exercised in unit tests without a ROS install. It models the
    two planar robots of the demo as points that must avoid a central square
    obstacle and each other.
    """

    def fn(joint_names, positions, mode):
        p = np.asarray(positions, dtype=float)
        # demo layout: 2 robots x 2 DOF (x, y) each
        if p.shape[0] != 4:
            return True
        a = p[0:2]
        b = p[2:4]
        # central square obstacle [-0.25, 0.25]^2
        for pt in (a, b):
            if np.all(np.abs(pt) < 0.25):
                return False
        # inter-robot clearance
        if np.linalg.norm(a - b) < 0.2:
            return False
        return True

    return CallableStateValidityChecker(fn)


def _demo_manip_analytic_checker(env: "MoveItEnvironment") -> CallableStateValidityChecker:
    """Manipulation-aware analytic checker for the headless pick-and-place demo.

    Like :func:`_demo_analytic_checker`, but it reads the mode's scene graph to
    locate movable objects: a movable object is positioned at its world anchor
    pose when resting, or follows the gripper (robot position + grasp offset) when
    held. A movable object collides with the central obstacle and with every robot
    except the one allowed to grasp it / currently holding it. This exercises the
    full attach/detach plumbing without a ROS install -- the real scene uses meshes
    via ``/check_state_validity``.
    """
    half = 0.25  # central obstacle half-size (matches center_box.stl)
    clearance = 0.2
    link_to_robot = {
        env.attach_links[r]: r for r in env.robots if env.attach_links[r] is not None
    }
    graspable_by = {}
    for t in env.tasks:
        manip = getattr(t, "manipulation", None)
        if manip is not None and manip.kind == "pick" and t.robots:
            graspable_by[manip.obj] = t.robots[0]

    def _robot_xy(positions, r):
        idx = env.robot_idx[r]
        return np.asarray(positions, dtype=float)[idx[0]: idx[-1] + 1][:2]

    def _in_center(xy):
        return abs(xy[0]) < half and abs(xy[1]) < half

    def fn(joint_names, positions, mode):
        rpos = {r: _robot_xy(positions, r) for r in env.robots}
        for xy in rpos.values():
            if _in_center(xy):
                return False
        rs = list(env.robots)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if np.linalg.norm(rpos[rs[i]] - rpos[rs[j]]) < clearance:
                    return False

        sg = mode.sg if mode is not None else {}
        for obj, value in sg.items():
            link, pose = value[0], value[1]
            if link == env.anchor_link:
                oxy = np.array(pose[:2], dtype=float)
                holder = None
            else:
                holder = link_to_robot.get(link)
                base = rpos.get(holder, np.zeros(2))
                oxy = base + np.array(pose[:2], dtype=float)
            if _in_center(oxy):
                return False
            for r, xy in rpos.items():
                if r == holder or r == graspable_by.get(obj):
                    continue
                if np.linalg.norm(oxy - xy) < clearance:
                    return False
        return True

    return CallableStateValidityChecker(fn)


def _register_demo_envs() -> None:
    if os.path.exists(_PICK_PLACE_SPEC):

        @register("moveit.pick_place_dependency")
        class _MoveItPickPlace(MoveItDependencyEnvironment):  # noqa: N801
            def __init__(self):
                spec = MoveItProblemSpec.from_file(_PICK_PLACE_SPEC)
                super().__init__(spec)
                self.set_validity_checker(_demo_manip_analytic_checker(self))

    if not os.path.exists(_DEMO_SPEC):
        return

    @register("moveit.demo_dependency")
    class _MoveItDemoDependency(MoveItDependencyEnvironment):  # noqa: N801
        def __init__(self):
            spec = MoveItProblemSpec.from_file(_DEMO_SPEC)
            super().__init__(spec, _demo_analytic_checker())

    @register("moveit.demo_sequence")
    class _MoveItDemoSequence(MoveItSequenceEnvironment):  # noqa: N801
        def __init__(self):
            spec = MoveItProblemSpec.from_file(_DEMO_SPEC)
            spec.mode_logic = "sequence"
            # A valid linear order consistent with the demo dependency graph.
            spec.sequence = [
                "a2_goal_0",
                "a2_goal_1",
                "a1_goal",
                "terminal",
            ]
            super().__init__(spec, _demo_analytic_checker())


_register_demo_envs()
