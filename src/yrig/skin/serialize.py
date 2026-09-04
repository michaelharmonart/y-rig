import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from yrig.io import confirm_overwrite
from yrig.shape import get_shape
from yrig.skin.core import (
    get_skin_cluster,
    get_skin_cluster_influences,
    get_skin_weights,
    set_skin_weights,
)

log = logging.getLogger(__name__)


@dataclass
class SkinWeightData:
    influences: list[str]
    skin_weights: dict[int, dict[str, float]]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        raw_weights = data["skin_weights"]
        skin_weights = {int(point_index): weights for point_index, weights in raw_weights.items()}
        return cls(influences=data["influences"], skin_weights=skin_weights)


def skin_weight_data_to_json(data: SkinWeightData) -> str:
    return json.dumps(data.to_dict(), indent=2)


def skin_weight_data_from_json(json_str: str) -> SkinWeightData:
    data = json.loads(json_str)
    return SkinWeightData.from_dict(data)


def skin_weight_data_from_file(filepath: Path) -> SkinWeightData:
    if not filepath.exists():
        raise FileNotFoundError
    with open(filepath) as file:
        serialized = file.read()
    return skin_weight_data_from_json(serialized)


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
    if not skin_cluster:
        resolved_skin_cluster = get_skin_cluster(geometry)
        if not resolved_skin_cluster:
            raise RuntimeError(f"No skinCluster on {geometry}")
    else:
        resolved_skin_cluster = skin_cluster

    if not confirm_overwrite(filepath, force):
        return False

    skin_weights = get_skin_weights(geometry, skin_cluster)
    influences = get_skin_cluster_influences(resolved_skin_cluster)
    skin_weight_data = SkinWeightData(influences=influences, skin_weights=skin_weights)
    serialized = skin_weight_data_to_json(skin_weight_data)
    with open(file=filepath, mode="w") as save_file:
        save_file.write(serialized)
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
    skin_weight_data = skin_weight_data_from_file(filepath)
    shape = get_shape(geometry)
    if shape is None:
        raise RuntimeError(f"{geometry} has no shape and can't be skinned.")
    applied_skin_cluster = set_skin_weights(
        shape, skin_weight_data.skin_weights, skin_cluster=skin_cluster
    )
    log.info(f"Skin weights applied to {applied_skin_cluster} from {filepath}")
    return applied_skin_cluster
