import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from yrig.build.nxt_api import execute_nxt_graph, nxt_file_roots
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
    build_path = rig_path / "data/build.nxt"
    resolved_scope = resolve_build_scope(build_scope)
    build_step = ProgressStep("Rig Build", callback=progress_callback)
    with nxt_file_roots(root_paths), bind_progress_step(build_step):
        try:
            no_components = resolved_scope == BuildScope.FACE
            execute_nxt_graph(build_path, parameters={"dev_build": dev_build})
        except Exception as e:
            build_logger.error("Build failed: %s", e)
            raise RuntimeError(f"mGear Shifter build failed for '{build_path}': {e}") from e
        return True
