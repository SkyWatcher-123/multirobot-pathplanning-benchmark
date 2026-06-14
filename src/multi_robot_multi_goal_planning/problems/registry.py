from typing import Callable, Dict
from multi_robot_multi_goal_planning.problems.planning_env import BaseProblem

EnvFactory = Callable[[], BaseProblem]
REGISTRY: Dict[str, EnvFactory] = {}


def register(names):
    """
    names can be:
    - a single string (simple case)
    - a list of (name, kwargs) pairs
    """

    def decorator(cls):
        if isinstance(names, str):
            REGISTRY[names] = lambda: cls()
        else:
            for name, kwargs in names:
                if name in REGISTRY:
                    raise ValueError(f"Duplicate env name: {name}")
                REGISTRY[name] = lambda cls=cls, kwargs=kwargs: cls(**kwargs)
        return cls

    return decorator

def get_all_environments():
    return REGISTRY

def _resolve_name(name: str) -> str:
    """Resolve an env name, tolerating '_' vs '.' separators.

    Environments are registered with dotted names (e.g. ``abstract.test``), but
    historically some call sites (and the original tests kept as examples) use the
    underscore form (``abstract_test``). When there is no exact match we fall back
    to a normalized match, but only against names that are actually registered --
    we never invent environments.
    """
    if name in REGISTRY:
        return name

    normalized = name.replace("_", ".")
    if normalized in REGISTRY:
        return normalized

    # Last resort: match ignoring the separator entirely, if unambiguous.
    flat = name.replace("_", "").replace(".", "")
    candidates = [k for k in REGISTRY if k.replace("_", "").replace(".", "") == flat]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(f"Unknown environment: {name}")


def get_env_by_name(name: str) -> BaseProblem:
    return REGISTRY[_resolve_name(name)]()


def list_envs(prefix: str = ""):
    return sorted([k for k in REGISTRY if k.startswith(prefix)])
