from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from yrig.io import promt_user_for_directory
from yrig.shape import get_shape
from yrig.skin.ng import has_ng_layer_data, write_ng_skin_data
from yrig.skin.serialize import export_skin_data

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

log = logging.getLogger(__name__)


def _resolve_export_directory(directory: Path | None = None) -> Path:
    if directory is not None:
        return directory
    resolved_directory = promt_user_for_directory()
    if resolved_directory is None:
        raise RuntimeError(
            "Unable to resolve skin export directory. Provide a directory or set asset root."
        )
    return resolved_directory


def _export_skin_data_for_shape(
    shape: str, directory: Path, filename: str, use_ng: bool, force: bool
) -> Path | None:
    use_ng_data = use_ng and has_ng_layer_data(shape)

    extension = ".json" if use_ng_data else ".yskin"
    filepath = directory / f"{filename}{extension}"

    if use_ng_data:
        result = write_ng_skin_data(filepath=filepath, geometry=shape, force=force)
    else:
        result = export_skin_data(filepath=filepath, geometry=shape, force=force)

    return filepath if result else None


def batch_export_skin_data(
    geometry: Iterable[str],
    directory: Path | None = None,
    *,
    use_ng: bool = True,
    force: bool = False,
    map_geo_to_file: Callable[[str], str] | None = None,
) -> dict[str, Path]:
    """Export skin weights for selected geometry or all skinned geometry in the scene.

    Args:
        directory: Optional directory to write skin files into.
        selected_only: If True, export only skin weights for currently selected geometry.
        use_ng: If True, export ngSkinTools JSON files (``.json``). If False, export
            yrig skin files (``.yskin``).
        force: When True, overwrite existing files without prompting.

    Returns:
        A list of paths for files that were written.
    """

    export_directory = _resolve_export_directory(directory)
    export_directory.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, Path] = {}
    for geo in geometry:
        shape = get_shape(geo)
        if shape is None:
            log.warning(f"No shape found for {geo}, skipping skin export.")
            continue
        filename = map_geo_to_file(geo) if map_geo_to_file else geo
        exported_path = _export_skin_data_for_shape(
            shape=shape, directory=export_directory, filename=filename, use_ng=use_ng, force=force
        )
        if exported_path is not None:
            exported_files[geo] = exported_path
    return exported_files
