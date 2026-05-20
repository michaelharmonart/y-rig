import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from maya import cmds

from yrig.build.progress import progress_step
from yrig.name import get_short_name, natural_sort_key
from yrig.skin.core import get_skin_clusters, skin_geometry
from yrig.skin.ng import apply_ng_skin_weights, get_influences_from_ng_skin_weights
from yrig.skin.serialize import apply_skin_weight_data, skin_weight_data_from_file

log = logging.getLogger(__name__)


def _resolve_valid_influences(influences: Iterable[str]) -> tuple[list[str], set[str]]:
    influence_names = [get_short_name(influence) for influence in influences]
    valid_influences = [influence for influence in influence_names if cmds.objExists(influence)]
    missing_influences = set(influence_names) - set(valid_influences)
    return valid_influences, missing_influences


def skin_and_apply_weights(filepath: Path, geometry: str) -> str:
    """
    Skin geometry (any type) using influences from a ``.yskin`` file and apply weights.

    Missing influences are skipped with a warning. Errors if no valid influences exist in the scene.
    """
    skin_weight_data = skin_weight_data_from_file(filepath)
    influence_names = skin_weight_data.influences
    valid_influences, missing_influences = _resolve_valid_influences(influence_names)
    if missing_influences:
        log.warning(
            f"[{geometry}] Missing {len(missing_influences)} influence(s) that were defined in its skin file : {sorted(missing_influences, key=natural_sort_key)}"
        )
    if not valid_influences:
        raise RuntimeError(
            f"The yskin file at {filepath} had no valid influences. Unable to skin geometry."
        )
    skin_cluster = skin_geometry(valid_influences, geometry)
    apply_skin_weight_data(skin_weight_data, geometry)
    log.info(f"Loaded yskin file for {geometry}")
    return skin_cluster


def skin_and_apply_ng_weights(filepath: Path, mesh: str) -> str:
    """
    Skin geometry using influences from an ngSkinTools file and apply weights.

    Missing influences are skipped with a warning. Errors if no valid influences exist in the scene.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} doesn't exist")
    influence_paths = get_influences_from_ng_skin_weights(filepath)
    valid_influences, missing_influences = _resolve_valid_influences(influence_paths)
    if missing_influences:
        log.warning(
            f"[{mesh}] Missing {len(missing_influences)} influence(s) that were defined in its skin file : {sorted(missing_influences, key=natural_sort_key)}"
        )

    # Only bind to joints specified in the skin file for final build
    if not valid_influences:
        raise RuntimeError(
            f"The ngskin file at {filepath} had no valid influences. Unable to skin geometry."
        )
    skin_cluster = skin_geometry(valid_influences, mesh)
    log.info(f"Skinned {mesh} to {len(valid_influences)} joint(s)")

    apply_ng_skin_weights(filepath, mesh)
    log.info(f"Loaded ng skin file for {mesh}")
    return skin_cluster


def skin_and_apply_weights_from_directory(
    directory: Path,
    geometry: Sequence[str],
    skip_skinned_geometry: bool = True,
    fallback_skinning: Callable[[str], Any] | None = None,
) -> None:
    """
    Skin geometry and apply saved weights from a directory.

    For each geometry name, loads either ``.json`` ngSkinTools weights or
    ``.yskin`` weights if present. If no weight file exists, optionally calls
    ``fallback_skinning``.
    """
    with progress_step("Skin Models") as progress:
        total = len(geometry)
        for i, geo in enumerate(geometry):
            if skip_skinned_geometry and get_skin_clusters(geo):
                continue
            ng_skin_manifest_filepath: Path = directory / f"{geo}/manifest.json"
            ng_skin_filepath: Path = directory / f"{geo}.json"
            yskin_filepath: Path = directory / f"{geo}.yskin"
            if ng_skin_manifest_filepath.exists():
                skin_and_apply_ng_weights(ng_skin_manifest_filepath, geo)
            elif ng_skin_filepath.exists():
                skin_and_apply_ng_weights(ng_skin_filepath, geo)
            elif yskin_filepath.exists():
                skin_and_apply_weights(yskin_filepath, geo)
            else:
                if fallback_skinning is not None:
                    fallback_skinning(geo)
            progress.update_progress((i + 1) / total)
