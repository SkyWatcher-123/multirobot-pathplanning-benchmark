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

        self.manipulating_env = False
        self._current_mode: Optional[Mode] = None
        self._display_fn: Optional[Callable[[List[State]], None]] = None

        self._validity_checker: StateValidityChecker = (
            validity_checker or _UnsetStateValidityChecker()
        )

        n_robots = len(self.robots)
        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT if n_robots > 1 else AgentType.SINGLE_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.STATIC,
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
        # Static scenes (the common case for MoveIt collision-checking problems)
        # have no scene graph. Manipulation problems can override this to return
        # attach/detach info that the ROS checker turns into AttachedCollisionObjects.
        return {}

    def set_to_mode(self, mode: Mode) -> None:
        # Remember the mode so the checker (and any attached-object handling)
        # can use it; nothing else to do for a static scene.
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
        """Convert a planned path into a DisplayTrajectory dict (ROS-independent)."""
        from multi_robot_multi_goal_planning.ros.trajectory_conversion import (
            to_display_trajectory_dict,
        )

        return to_display_trajectory_dict(
            path,
            self.joint_names,
            base_frame=self.base_frame,
            velocity=self.velocity,
            split_by_mode=split_by_mode,
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


def _register_demo_envs() -> None:
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
