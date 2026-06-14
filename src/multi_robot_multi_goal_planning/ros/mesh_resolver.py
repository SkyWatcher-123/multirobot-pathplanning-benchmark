"""Resolve mesh URIs (``package://``, ``file://``, plain paths) to filesystem paths.

Kept ROS-optional: ``package://`` resolution uses ``rospkg`` when available (the
normal ROS case) and otherwise falls back to the ``ROS_PACKAGE_PATH`` environment
variable, so the helper is unit-testable without a full ROS install.
"""

from __future__ import annotations

import os


def resolve_mesh_path(uri: str) -> str:
    """Return a local filesystem path for a mesh URI.

    Supports ``package://<pkg>/<rel>``, ``file://<abs>`` and plain paths.
    """
    if uri.startswith("file://"):
        return uri[len("file://") :]

    if uri.startswith("package://"):
        rest = uri[len("package://") :]
        pkg, _, rel = rest.partition("/")
        pkg_dir = _find_package_dir(pkg)
        if pkg_dir is None:
            raise FileNotFoundError(
                f"could not locate ROS package '{pkg}' to resolve mesh '{uri}'. "
                f"Make sure the package is on ROS_PACKAGE_PATH / your catkin "
                f"workspace is sourced."
            )
        return os.path.join(pkg_dir, rel)

    return uri


def _find_package_dir(pkg: str):
    # Preferred: rospkg (handles a sourced catkin workspace correctly).
    try:
        import rospkg  # type: ignore

        return rospkg.RosPack().get_path(pkg)
    except Exception:
        pass

    # Fallback: scan ROS_PACKAGE_PATH for a directory containing the package
    # (identified by a package.xml with a matching <name>).
    for root in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
        if not root:
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if "package.xml" in filenames and os.path.basename(dirpath) == pkg:
                return dirpath
    return None
