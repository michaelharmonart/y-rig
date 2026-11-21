from collections.abc import Callable, Sequence
from pathlib import Path

from yrig.build.scope import BuildScope


def build_from_path(
    root_paths: Sequence[Path],
    rig_path: Path,
    dev_build: bool = False,
    build_scope: BuildScope | str | None = None,
    progress_callback: Callable[[float, str | None], None] | None = None,
) -> bool:
    """Backward-compatible entry point for the Lantern pipeline."""
    from yrig.build import build_rig

    return build_rig(
        root_paths=root_paths,
        rig_path=rig_path,
        dev_build=dev_build,
        build_scope=build_scope,
        progress_callback=progress_callback,
    )
