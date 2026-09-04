import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from maya import cmds

from yrig.build.progress import progress_step
from yrig.name import get_short_name, natural_sort_key
from yrig.skin.core import get_skin_clusters, skin_geometry
from yrig.skin.ng import apply_ng_skin_weights, get_influences_from_ng_skin_weights
from yrig.skin.serialize import apply_skin_weight_data, skin_weight_data_from_file

log = logging.getLogger(__name__)


def _valid_influences(
    influence_names: Iterable[str],
    geometry: str,
    filepath: Path,
) -> list[str]:
    valid = [name for name in influence_names if cmds.objExists(name)]
    missing = set(influence_names) - set(valid)
    if missing:
        log.warning(
            f"[{geometry}] Missing {len(missing)} influence(s) that were defined in its skin file: "
            f"{sorted(missing, key=natural_sort_key)}"
        )
    if not valid:
        raise RuntimeError(
            f"The file at {filepath} had no valid influences. Unable to skin geometry."
        )
    return valid


def apply_weights(filepath: Path, geometry: str) -> None:
    """
    Apply weights from a ``.yskin`` file.

    Missing influences are skipped with a warning. Errors if no valid
    influences exist in the scene.
    """
    skin_weight_data = skin_weight_data_from_file(filepath)
    valid_influences = _valid_influences(
        skin_weight_data.influences,
        geometry,
        filepath,
    )
    apply_skin_weight_data(skin_weight_data, geometry)


def skin_and_apply_weights(filepath: Path, geometry: str) -> str:
    """
    Skin geometry using influences from a ``.yskin`` file and apply weights.
    """
    skin_weight_data = skin_weight_data_from_file(filepath)
    valid_influences = _valid_influences(
        skin_weight_data.influences,
        geometry,
        filepath,
    )

    skin_cluster = skin_geometry(valid_influences, geometry)
    apply_skin_weight_data(skin_weight_data, geometry)

    log.info(f"Loaded yskin file for {geometry} from {filepath}")
    return skin_cluster


def apply_ng_weights(filepath: Path, mesh: str) -> None:
    """
    Apply weights from a ngSkinTools file.

    Missing influences are skipped with a warning. Errors if no valid
    influences exist in the scene.
    """
    skin_weight_data = skin_weight_data_from_file(filepath)
    valid_influences = _valid_influences(
        skin_weight_data.influences,
        mesh,
        filepath,
    )
    apply_ng_skin_weights(filepath, mesh)


def skin_and_apply_ng_weights(filepath: Path, mesh: str) -> str:
    """
    Skin geometry using influences from an ngSkinTools file and apply weights.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} doesn't exist")

    influence_paths = get_influences_from_ng_skin_weights(filepath)
    influence_names = [get_short_name(path) for path in influence_paths]

    valid_influences = _valid_influences(
        influence_names,
        mesh,
        filepath,
    )

    skin_cluster = skin_geometry(valid_influences, mesh)
    log.info(f"Skinned {mesh} to {len(valid_influences)} joint(s)")

    apply_ng_skin_weights(filepath, mesh)
    log.info(f"Loaded ng skin file for {mesh}")

    return skin_cluster


def skin_and_apply_weights_from_directories(
    directories: Sequence[Path],
    geometry: Sequence[str],
    skip_skinned_geometry: bool = True,
    fallback_skinning: Callable[[str], Any] | None = None,
    map_geo_to_file: Callable[[str], str] | None = None,
) -> None:
    """
    Skin geometry and apply saved weights from one or more directories.
    For each geometry, searches the directories for a matching ``.json`` or ``.yskin`` weight file.
    If no weight file is found, optionally calls ``fallback_skinning``.

    Args:
        directories: Directories to search for weight files, in search order.
        geometry: Geometry to skin and apply weights to.
        skip_skinned_geometry: Whether to skip geometry that already has a skin cluster.
        fallback_skinning: Optional function called for geometry with no saved weight file.
        map_geo_to_file: Optional function that maps a geometry name to the corresponding weight file name.
    """
    with progress_step("Skin Models", total=len(geometry)) as progress:
        for geo in geometry:
            with progress_step(geo):
                if skip_skinned_geometry and get_skin_clusters(geo):
                    continue
                skinned: bool = False
                for directory in directories:
                    geo_mapped_file = map_geo_to_file(geo) if map_geo_to_file is not None else geo
                    ng_skin_filepath: Path = directory / f"{geo_mapped_file}.json"
                    yskin_filepath: Path = directory / f"{geo_mapped_file}.yskin"
                    if ng_skin_filepath.exists():
                        skin_and_apply_ng_weights(ng_skin_filepath, geo)
                        skinned = True
                        break
                    elif yskin_filepath.exists():
                        skin_and_apply_weights(yskin_filepath, geo)
                        skinned = True
                        break
                if not skinned and fallback_skinning is not None:
                    fallback_skinning(geo)
