import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from maya import cmds

from yrig.io import confirm_overwrite
from yrig.io.json import export_json, load_json
from yrig.maya_api.enum import (
    SkinClusterNormalizeWeights,
    SkinClusterRelativeSpaceMode,
    SkinClusterSkinningMethod,
    SkinClusterWeightDistribution,
)
from yrig.maya_api.node import SkinCluster
from yrig.name import natural_sort_key
from yrig.shape import get_shape
from yrig.skin.core import (
    get_skin_cluster,
    get_skin_cluster_influences,
    get_skin_weights,
    set_skin_weights,
    skin_geometry,
)

log = logging.getLogger(__name__)


@dataclass
class SkinBindData:
    name: str
    influences: list[str]
    skinning_method: SkinClusterSkinningMethod
    relative_space_mode: SkinClusterRelativeSpaceMode
    support_non_rigid: bool
    normalize_weights: SkinClusterNormalizeWeights
    weight_distribution: SkinClusterWeightDistribution
    max_influences: int
    maintain_max_influences: bool


@dataclass
class SkinWeightData:
    influences: list[str]
    skin_weights: dict[int, dict[str, float]]


def _validate_influences(
    influence_names: Iterable[str],
    geometry: str,
    error_on_missing: bool = False,
) -> list[str]:
    valid = [name for name in influence_names if cmds.objExists(name)]
    missing = set(influence_names) - set(valid)
    if missing:
        missing_message = (
            f"[{geometry}] Missing {len(missing)} influence(s) that were defined in its bind data: "
            f"{sorted(missing, key=natural_sort_key)}"
        )
        if error_on_missing:
            raise RuntimeError(missing_message)
        else:
            log.warning(missing_message)
    if not valid:
        raise RuntimeError("No valid influences. Unable to skin geometry.")
    return valid


def load_skin_bind_data(filepath: Path) -> SkinBindData:
    return load_json(filepath, SkinBindData)


def export_skin_bind_data(
    filepath: Path,
    skin_cluster: str | SkinCluster,
    force: bool = False,
) -> bool:

    if filepath.suffix != ".ybind":
        raise ValueError("Skin bind files should use the .ybind extension.")
    bind_data = get_skin_bind_data(skin_cluster)
    if not confirm_overwrite(filepath, force):
        return False
    export_json(filepath, bind_data)
    log.info(f"The skin bind data for {skin_cluster} was written to {filepath}")
    return True


def get_skin_bind_data(skin_cluster: str | SkinCluster) -> SkinBindData:
    skin_cluster_node = (
        skin_cluster
        if isinstance(skin_cluster, SkinCluster)
        else SkinCluster.from_existing(skin_cluster)
    )
    return SkinBindData(
        name=str(skin_cluster),
        influences=get_skin_cluster_influences(skin_cluster),
        skinning_method=skin_cluster_node.skinning_method.get(),
        relative_space_mode=skin_cluster_node.relative_space_mode.get(),
        support_non_rigid=skin_cluster_node.dqs_support_non_rigid.get(),
        normalize_weights=skin_cluster_node.normalize_weights.get(),
        weight_distribution=skin_cluster_node.weight_distribution.get(),
        max_influences=skin_cluster_node.max_influences.get(),
        maintain_max_influences=skin_cluster_node.maintain_max_influences.get(),
    )


def apply_skin_bind_data(skin_cluster: str | SkinCluster, data: SkinBindData) -> None:
    skin_cluster_node = (
        skin_cluster
        if isinstance(skin_cluster, SkinCluster)
        else SkinCluster.from_existing(skin_cluster)
    )
    skin_cluster_node.skinning_method.set(data.skinning_method)
    skin_cluster_node.relative_space_mode.set(data.relative_space_mode)
    skin_cluster_node.dqs_support_non_rigid.set(data.support_non_rigid)
    skin_cluster_node.normalize_weights.set(data.normalize_weights)
    skin_cluster_node.weight_distribution.set(data.weight_distribution)
    skin_cluster_node.max_influences.set(data.max_influences)
    skin_cluster_node.maintain_max_influences.set(data.maintain_max_influences)


def skin_geometry_from_bind_data(
    geometry: str, data: SkinBindData, skip_missing_influences: bool = True
) -> SkinCluster:
    valid_influences = _validate_influences(
        data.influences, geometry, error_on_missing=not skip_missing_influences
    )
    return skin_geometry(
        bind_joints=valid_influences,
        geometry=geometry,
        name=data.name,
        dual_quaternion=data.skinning_method
        in {SkinClusterSkinningMethod.DUAL_QUATERNION, SkinClusterSkinningMethod.WEIGHT_BLENDED},
        weight_blend=data.skinning_method == SkinClusterSkinningMethod.WEIGHT_BLENDED,
        support_non_rigid=data.support_non_rigid,
        relative_space_mode=data.relative_space_mode,
        normalize_weights=data.normalize_weights,
        weight_distribution=data.weight_distribution,
        max_influences=data.max_influences,
        maintain_max_influences=data.maintain_max_influences,
    )


def load_skin_weight_data(filepath: Path) -> SkinWeightData:
    return load_json(filepath, SkinWeightData)


def apply_skin_weight_data(
    data: SkinWeightData, geometry: str, skin_cluster: str | None = None
) -> str:
    """
    Apply SkinWeightData to the skinCluster on the given geometry.

    Args:
        data: SkinWeightData object.
        geometry: Target mesh or transform to apply weights to.
        skin_cluster: Optional specification of which skinCluster node.

    Returns:
        str: The name of the skinCluster that the weights were applied to.
    """
    shape = get_shape(geometry)
    if shape is None:
        raise RuntimeError(f"{geometry} has no attached shape node")
    applied_skin_cluster = set_skin_weights(shape, data.skin_weights, skin_cluster=skin_cluster)
    return applied_skin_cluster


def get_skin_weight_data(geometry: str, skin_cluster: str | None = None) -> SkinWeightData:
    if not skin_cluster:
        resolved_skin_cluster = get_skin_cluster(geometry)
        if not resolved_skin_cluster:
            raise RuntimeError(f"No skinCluster on {geometry}")
    else:
        resolved_skin_cluster = skin_cluster
    skin_weights = get_skin_weights(geometry, skin_cluster)
    influences = get_skin_cluster_influences(resolved_skin_cluster)
    return SkinWeightData(influences=influences, skin_weights=skin_weights)


def export_skin_weights(
    filepath: Path, geometry: str, skin_cluster: str | None = None, force: bool = False
) -> bool:
    """
    Export skin weights from a geometry's skinCluster to a file.

    The output file will be JSON, but should have the `.yskin` extension.

    Args:
        filepath: Destination path (should use `.yskin` extension).
        geometry: Mesh or transform containing the skinned geometry.
        skin_cluster: Optional specification of which skinCluster node.
        force: If True, overwrite existing files without prompting.

    Returns:
        True if export succeeded, False if aborted due to overwrite check.
    """
    if filepath.suffix != ".yskin":
        raise ValueError("Skin weight files should use the .yskin extension.")
    if not skin_cluster:
        resolved_skin_cluster = get_skin_cluster(geometry)
        if not resolved_skin_cluster:
            raise RuntimeError(f"No skinCluster on {geometry}")
    else:
        resolved_skin_cluster = skin_cluster
    if not confirm_overwrite(filepath, force):
        return False
    skin_weight_data = get_skin_weight_data(geometry, resolved_skin_cluster)
    export_json(filepath, skin_weight_data)
    log.info(f"The skin weights for {resolved_skin_cluster} were written to {filepath}")
    return True


def import_skin_weights(filepath: Path, geometry: str, skin_cluster: str | None = None) -> str:
    """
    Import skin weights from a file and apply them to the skinCluster on the given geometry.
    The input file must be a `.yskin` JSON-based skin weight file produced by yrig.

    Args:
        filepath: Path to `.yskin` skin weight file.
        geometry: Target mesh or transform to apply weights to.
        skin_cluster: Optional specification of which skinCluster node.

    Raises:
        FileNotFoundError: If the `.yskin` file does not exist.
        RuntimeError: If geometry has no valid shape node or cannot be resolved.

    Returns:
        str: The name of the skinCluster that the weights were applied to.
    """
    skin_weight_data = load_skin_weight_data(filepath)
    shape = get_shape(geometry)
    if shape is None:
        raise RuntimeError(f"{geometry} has no shape and can't be skinned.")
    applied_skin_cluster = set_skin_weights(
        shape, skin_weight_data.skin_weights, skin_cluster=skin_cluster
    )
    log.info(f"Skin weights applied to {applied_skin_cluster} from {filepath}")
    return applied_skin_cluster


def export_skin_data(
    filepath: Path,
    geometry: str,
    skin_cluster: str | None = None,
    force: bool = False,
) -> bool:
    """Export skin binding and weights to ``.ybind`` and ``.yskin`` files."""
    if skin_cluster is None:
        skin_cluster = get_skin_cluster(geometry)
        if not skin_cluster:
            raise RuntimeError(f"No skinCluster on {geometry}")

    weight_filepath = filepath
    bind_filepath = filepath.with_suffix(".ybind")

    if not confirm_overwrite((weight_filepath, bind_filepath), force):
        return False

    bind_data = get_skin_bind_data(skin_cluster)
    weight_data = get_skin_weight_data(geometry, skin_cluster)

    export_json(bind_filepath, bind_data)
    export_json(weight_filepath, weight_data)

    log.info(
        f"The skin binding and weights for {geometry} were written to {filepath.with_suffix('')} "
        f"({bind_filepath.suffix} and {weight_filepath.suffix})"
    )
    return True
