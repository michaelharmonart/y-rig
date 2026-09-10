from collections.abc import Collection, Iterable
from pathlib import Path

from yrig.io import confirm_overwrite
from yrig.io.json import export_json, load_json
from yrig.maya_api.node import BlendShape

from .apply import apply_blendshape_data
from .data import BlendShapeData
from .get import get_blendshape_data


def import_blendshape(
    filepath: Path,
    blendshape: str | BlendShape,
    directories: Collection[str] | None = None,
    targets: Collection[str] | None = None,
    parent_directory: str | None = None,
) -> None:
    """
    Import blendShape target data from a `.yshape` file.

    Args:
        filepath: Source `.yshape` file containing serialized blendShape data.
        blendshape: BlendShape node to modify.
        directories: Specify target directories to import.
        targets: Specify target names to import.
    """
    data = load_json(filepath, BlendShapeData)
    apply_blendshape_data(blendshape, data, directories, targets)


def export_blendshape(
    filepath: Path,
    blendshape: str | BlendShape,
    directories: Collection[str] | None = None,
    targets: Iterable[str | int] | None = None,
    force: bool = False,
) -> bool:
    """
    Export blendShape target data to a `.yshape` file.

    Args:
        filepath: Destination `.yshape` file.
        blendshape: BlendShape node to export.
        directories: Specify target directories to import.
        targets: Specify target names or indices to export.
        force: Whether to overwrite an existing file without confirmation.

    Returns:
        ``True`` if the blendShape was exported, or ``False``
        if the export was cancelled."""
    if filepath.suffix != ".yshape":
        raise ValueError("Blendshaspe files should use the .yshape extension.")
    if not confirm_overwrite(filepath, force):
        return False
    blendshape_data = get_blendshape_data(blendshape, directories, targets)
    export_json(filepath, blendshape_data)
    return True
