import logging
from collections.abc import Callable, Collection, Iterable, Sequence
from pathlib import Path

from yrig.build.progress import progress_step
from yrig.deformer.blendshape.core import connect_to_psd_network
from yrig.io import confirm_overwrite
from yrig.io.json import export_json, load_json
from yrig.maya_api.node import BlendShapeNode
from yrig.shape import get_shape

from .apply import apply_blendshape_data
from .data import BlendShapeData
from .get import get_blendshape_data

log = logging.getLogger(__name__)


def import_blendshape(
    filepath: Path,
    blendshape: str | BlendShapeNode,
    *,
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
    apply_blendshape_data(
        blendshape,
        data,
        directories=directories,
        targets=targets,
        parent_directory=parent_directory,
    )
    log.info(f"Imported shapes from {filepath}")


def export_blendshape(
    filepath: Path,
    blendshape: str | BlendShapeNode,
    *,
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
        parent_directory: Parent directory for the specified directories and targets,
            or the directory to parent the entire imported structure on if neither are specified.
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
    log.info(f"Exported shapes to {filepath}")
    return True


def import_blendshapes_from_directories(
    directories: Sequence[Path],
    geometry: Sequence[str],
    geo_to_blendshape: Callable[[str], str | BlendShapeNode],
    *,
    geo_to_filename: Callable[[str], str] | None = None,
    psd_network: str | None = None,
) -> dict[str, Path | None]:
    geo_applied_files: dict[str, Path | None] = {}
    with progress_step("Import Blendshapes", total=len(geometry)):
        for geo in geometry:
            with progress_step(geo):
                applied: bool = False
                geo_mapped_file = geo_to_filename(geo) if geo_to_filename is not None else geo
                blendshape = geo_to_blendshape(geo)
                for directory in directories:
                    yshape_filepath: Path = directory / f"{geo_mapped_file}.yshape"
                    if yshape_filepath.exists():
                        import_blendshape(yshape_filepath, blendshape)
                        geo_applied_files[geo] = yshape_filepath
                        applied = True
                        if psd_network:
                            connect_to_psd_network(blendshape, psd_network)
                        break
                if not applied:
                    log.warning(
                        f"Couldn't find a .yshape file named '{geo_mapped_file}' for {geometry} in any of the checked directories: {directories}"
                    )
    return geo_applied_files


def batch_export_blendshapes(
    geometry: Iterable[str],
    directory: Path,
    geo_to_blendshape: Callable[[str], str | BlendShapeNode],
    *,
    force: bool = False,
    geo_to_filename: Callable[[str], str] | None = None,
    directories: Collection[str] | None = None,
    targets: Iterable[str | int] | None = None,
) -> dict[str, Path]:

    export_directory = directory
    export_directory.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, Path] = {}
    for geo in geometry:
        shape = get_shape(geo)
        if shape is None:
            log.warning(f"No shape found for {geo}, skipping blendshape export.")
            continue
        filename = geo_to_filename(geo) if geo_to_filename else geo
        filepath = export_directory / f"{filename}.yshape"
        blendshape = geo_to_blendshape(geo)
        export_blendshape(
            filepath, blendshape, force=force, directories=directories, targets=targets
        )
        exported_files[geo] = filepath
    return exported_files
