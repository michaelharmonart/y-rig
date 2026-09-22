import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from yrig.build.progress import progress_step
from yrig.maya_api.node import SkinCluster
from yrig.name import get_short_name
from yrig.skin.core import _resolve_skin_cluster, get_skin_clusters, skin_geometry
from yrig.skin.ng import apply_ng_skin_weights, get_influences_from_ng_skin_weights
from yrig.skin.serialize import (
    SkinBindData,
    _validate_influences,
    apply_skin_bind_data,
    apply_skin_weight_data,
    load_skin_bind_data,
    load_skin_weight_data,
    skin_geometry_from_bind_data,
)

log = logging.getLogger(__name__)


def _get_bind_data(weights_filepath: Path) -> SkinBindData | None:
    bind_filepath = weights_filepath.with_suffix(".ybind")
    if bind_filepath.exists():
        return load_skin_bind_data(bind_filepath)
    else:
        return None


def apply_skin_data(filepath: Path, geometry: str) -> None:
    """
    Apply weights from a ``.yskin`` file and bind data from a ``.ybind` file
    """
    skin_cluster = _resolve_skin_cluster(geometry)
    if skin_cluster is None:
        raise RuntimeError(f"Couldn't find a skinCluster on {geometry}")
    apply_weights(filepath, geometry)
    bind_data = _get_bind_data(filepath)
    if bind_data is not None:
        apply_skin_bind_data(skin_cluster, bind_data)


def apply_weights(filepath: Path, geometry: str) -> None:
    """
    Apply weights from a ``.yskin`` file.

    Missing influences are skipped with a warning. Errors if no valid
    influences exist in the scene.
    """
    skin_weight_data = load_skin_weight_data(filepath)
    valid_influences = _validate_influences(
        skin_weight_data.influences,
        geometry,
    )
    apply_skin_weight_data(skin_weight_data, geometry)


def skin_and_apply_weights(filepath: Path, geometry: str) -> SkinCluster:
    """
    Skin geometry using influences from a ``.yskin`` file and apply weights.
    """
    skin_weight_data = load_skin_weight_data(filepath)
    skin_bind_data = _get_bind_data(filepath)
    if skin_bind_data is not None:
        skin_geometry_from_bind_data(geometry, skin_bind_data)
    else:
        valid_influences = _validate_influences(
            skin_weight_data.influences,
            geometry,
        )
        skin_cluster = skin_geometry(valid_influences, geometry)
    apply_skin_weight_data(skin_weight_data, geometry)

    log.info(f"Loaded yskin file for {geometry} from {filepath}")
    return skin_cluster


def apply_ng_weights(filepath: Path, mesh: str) -> None:
    """
    Apply weights from a ngSkinTools file.
    """
    apply_ng_skin_weights(filepath, mesh)


def apply_ng_data(filepath: Path, mesh: str) -> None:
    """
    Apply weights from a ngSkinTools file and apply bind data if present.
    """
    skin_cluster = _resolve_skin_cluster(mesh)
    if skin_cluster is None:
        raise RuntimeError(f"Couldn't find a skinCluster on {mesh}")
    apply_ng_skin_weights(filepath, mesh)
    bind_data = _get_bind_data(filepath)
    if bind_data is not None:
        apply_skin_bind_data(skin_cluster, bind_data)


def skin_and_apply_ng_weights(filepath: Path, mesh: str) -> SkinCluster:
    """
    Skin geometry using influences from an ngSkinTools file and apply weights.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} doesn't exist")
    skin_bind_data = _get_bind_data(filepath)
    if skin_bind_data is not None:
        skin_cluster = skin_geometry_from_bind_data(mesh, skin_bind_data)
    else:
        influence_paths = get_influences_from_ng_skin_weights(filepath)
        influence_names = [get_short_name(path) for path in influence_paths]

        valid_influences = _validate_influences(
            influence_names,
            mesh,
        )

        skin_cluster = skin_geometry(valid_influences, mesh)
        log.info(f"Skinned {mesh} to {len(valid_influences)} joint(s)")

    apply_ng_skin_weights(filepath, mesh)
    log.info(f"Loaded ng skin file for {mesh} from {filepath}")

    return skin_cluster


def apply_skin_data_from_directories(
    directories: Sequence[Path],
    geometry: Sequence[str],
    *,
    map_geo_to_file: Callable[[str], str] | None = None,
) -> dict[str, Path]:
    """
    Apply saved weights from one or more directories.
    For each geometry, searches the directories for a matching ``.json`` or ``.yskin`` weight file.

    Args:
        directories: Directories to search for weight files, in search order.
        geometry: Geometry to skin and apply weights to.
        skip_skinned_geometry: Whether to skip geometry that already has a skin cluster.
        map_geo_to_file: Optional function that maps a geometry name to the corresponding weight file name.

    Returns:
        Dictionary mapping geometry that had skin weights applied -> the skin weight file applied.
    """
    geo_applied_weight_files: dict[str, Path] = {}
    with progress_step("Apply Skin Weights", total=len(geometry)) as progress:
        for geo in geometry:
            with progress_step(geo):
                if not get_skin_clusters(geo):
                    log.warning(
                        f"Specified geometry {geo} didn't have a skinCluster and couldn't have weights applied."
                    )
                applied: bool = False
                for directory in directories:
                    geo_mapped_file = map_geo_to_file(geo) if map_geo_to_file is not None else geo
                    ng_skin_filepath: Path = directory / f"{geo_mapped_file}.json"
                    yskin_filepath: Path = directory / f"{geo_mapped_file}.yskin"
                    if ng_skin_filepath.exists():
                        apply_ng_data(ng_skin_filepath, geo)
                        geo_applied_weight_files[geo] = ng_skin_filepath
                        applied = True
                        break
                    elif yskin_filepath.exists():
                        apply_skin_data(yskin_filepath, geo)
                        geo_applied_weight_files[geo] = yskin_filepath
                        applied = True
                        break
                if not applied:
                    log.warning(
                        f"Couldn't find a skin file named '{geo_mapped_file}' for {geometry} in any of the checked directories: {directories}"
                    )
    return geo_applied_weight_files


def skin_and_apply_weights_from_directories(
    directories: Sequence[Path],
    geometry: Sequence[str],
    *,
    skip_skinned_geometry: bool = True,
    fallback_skinning: Callable[[str], Any] | None = None,
    map_geo_to_file: Callable[[str], str] | None = None,
) -> dict[str, Path | None]:
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

    Returns:
        Returns:
            Dictionary mapping geometry that was skinned -> the skin weight file applied or None if default skinning.
    """
    geo_applied_weight_files: dict[str, Path | None] = {}
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
                        geo_applied_weight_files[geo] = ng_skin_filepath
                        skinned = True
                        break
                    elif yskin_filepath.exists():
                        skin_and_apply_weights(yskin_filepath, geo)
                        geo_applied_weight_files[geo] = yskin_filepath
                        skinned = True
                        break
                if not skinned:
                    if fallback_skinning is not None:
                        fallback_skinning(geo)
                        geo_applied_weight_files[geo] = None
                    else:
                        log.warning(
                            f"Couldn't find a skin file named '{geo_mapped_file}' for {geometry} in any of the checked directories: {directories}"
                        )
    return geo_applied_weight_files
