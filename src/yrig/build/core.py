import logging
from collections.abc import Callable, Sequence
from itertools import chain
from pathlib import Path

from yrig.build.nxt_api import YRIG_NXT_DIR, execute_nxt_graph, nxt_file_roots
from yrig.build.progress import ProgressStep, bind_progress_step
from yrig.build.scope import BuildScope

build_logger = logging.getLogger("yrig.build")


def resolve_build_scope(value: str | BuildScope | None) -> BuildScope | None:
    if value is None:
        return None
    if isinstance(value, BuildScope):
        return value
    try:
        return BuildScope(value)
    except ValueError:
        valid = [e.value for e in BuildScope]
        build_logger.error("Invalid BuildScope '%s'. Valid options: %s", value, valid)
        raise


def build_rig(
    root_paths: Sequence[Path],
    rig_path: Path,
    dev_build: bool = False,
    build_scope: BuildScope | str | None = None,
    progress_callback: Callable[[float, str | None], None] | None = None,
) -> bool:
    """Build a y-rig rig.

    Args:
        root_paths: Paths to rig build root directories. Should be in order of strongest -> weakest.
        rig_path: Path to the rig's build directory relative to the root. For example characters/rig_name
        dev_build: When true the rig build will not have working data stripped for final export.
        progress_callback: A function to call at each step of the build for progress monitoring.
            It will be called with a float (overall progress from 0-1) and a string (the current step)
        build_scope: An optional BuildScope which will limit the build to that scope.

    Returns:
        bool: True if the build was successful, else False
    """
    extended_root_paths = chain(root_paths, (YRIG_NXT_DIR,))
    build_path = rig_path / "data/build.nxt"
    resolved_scope = resolve_build_scope(build_scope)
    build_step = ProgressStep("Rig Build", callback=progress_callback)
    with nxt_file_roots(extended_root_paths), bind_progress_step(build_step):
        try:
            execute_nxt_graph(
                build_path, parameters={"/.dev_build": dev_build, "/.build_scope": resolved_scope}
            )
        except Exception as e:
            build_logger.error("Build failed: %s", e)
            raise RuntimeError(f"y-rig build failed for '{build_path}': {e}") from e
    return True
