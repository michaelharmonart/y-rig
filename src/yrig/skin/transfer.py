from maya import cmds

from yrig.skin.core import (
    _add_missing_influences,
    _resolve_skin_cluster,
    get_skin_cluster_influences,
    skin_geometry,
)
from yrig.skin.serialize import apply_skin_bind_data, get_skin_bind_data


def transfer_skin(
    source: str,
    target: str,
    interpolate: bool = True,
    map_by_name: bool = True,
    add_missing_influences: bool = True,
    skin_unskinned_target: bool = True,
) -> None:
    """Transfer skin weights from one skinned object to another.

    Args:
        source: Source geometry or skinCluster.
        target: Target geometry or skinCluster.
        interpolate: Smooth/interpolate weights when transferring between different topology.
        map_by_name: Match influences by name. Otherwise Maya uses closestJoint.
        add_missing_influences: Add source influences that do not exist on the target skinCluster.
        skin_unskinned_target: Create a skinCluster on the target when one does not exist.
    """
    source_skin = _resolve_skin_cluster(source)
    if source_skin is None:
        raise RuntimeError(f"No skin cluster found on {source}.")

    source_influences = get_skin_cluster_influences(source_skin)
    if not source_influences:
        raise RuntimeError(f"Source skin cluster {source_skin!r} has no influences.")

    target_skin = _resolve_skin_cluster(target)

    if target_skin is None:
        if skin_unskinned_target:
            target_skin = str(skin_geometry(source_influences, target))
        else:
            raise RuntimeError(f"No skin cluster found on {target}.")
    elif add_missing_influences:
        target_influences = get_skin_cluster_influences(target_skin)
        _add_missing_influences(target_influences, source_influences, str(target_skin))

    cmds.copySkinWeights(
        sourceSkin=str(source_skin),
        destinationSkin=str(target_skin),
        noMirror=True,
        smooth=interpolate,
        influenceAssociation="name" if map_by_name else "closestJoint",
    )
    bind_data = get_skin_bind_data(source_skin)
    apply_skin_bind_data(target_skin, bind_data)
