from collections.abc import Iterable

from yrig.shape import get_shape
from yrig.skin.core import get_skin_cluster
from yrig.skin.ng import (
    get_ng_layer,
    get_ng_layer_used_influence_index_to_name_mapping,
    get_ng_layer_weights,
    set_ng_layer_weights,
)

from .data import WeightSplitData
from .operations import compute_split_weights
from .tag import get_weight_split_data_from_influences


def split_ng_layer_weights(
    mesh: str,
    *,
    split_data_collection: Iterable[WeightSplitData] | None = None,
    layer: str | None = None,
    skin_cluster: str | None = None,
) -> None:
    """
    .. warning::
        This function is BUNS SLOW since it has to call an ngSkinTools command for every influence
        in the layer once to get the weights, and once to set the new split ones.

    This function is designed to reassign weights from a set of original joints (e.g., proxy drivers)
    across multiple split joints (e.g., spline-based deformation chains like ribbons or bendy limbs).
    The redistribution is done by computing weights along a spline built from the split joints'
    world positions and distributing the original joint's influence accordingly.

    For each `WeightSplitData` entry a temporary NURBS curve is built from the
    world-space positions of the split influences. Every vertex that is affected by the
    source influence is projected onto that curve and assigned new weights via B-spline
    basis evaluation. The source influence's weight is then zeroed out and its value is
    redistributed across the split influences proportionally.

    Args:
        mesh: The transform node or mesh shape.
        split_data_collection: One or more `WeightSplitData` descriptors, each
            specifying a source influence and the ordered list of split
            influences that should receive its weights.  The ``degree`` and ``periodic``
            fields on each descriptor control the spline used for interpolation.
        layer: The name of the layer to split. If None, the active layer will be used.
        skin_cluster: Explicit skinCluster node name to operate on.  When ``None``
            the first skinCluster found on *mesh* is used.

    Raises:
        RuntimeError: If no skinCluster can be resolved for *mesh*.
    """
    # get the shape node
    mesh_shape = get_shape(mesh)
    if mesh_shape is None:
        raise RuntimeError(f"{mesh} has no attached shape node")
    # get the skinCluster and weights
    resolved_skin_cluster = skin_cluster if skin_cluster is not None else get_skin_cluster(mesh)
    if resolved_skin_cluster is None:
        raise RuntimeError(f"Coudn't find a skinCluster on {mesh}.")
    resolved_layer = get_ng_layer(resolved_skin_cluster, layer)
    if resolved_layer is None:
        raise RuntimeError(
            f"Coudn't find the specified layer ({layer}) on {resolved_skin_cluster}."
        )
    if split_data_collection is not None:
        resolved_split_data_collection = list(split_data_collection)
    else:
        used_influences = get_ng_layer_used_influence_index_to_name_mapping(
            resolved_layer, resolved_skin_cluster
        )
        weight_split_data_list = get_weight_split_data_from_influences(used_influences.values())
        resolved_split_data_collection = weight_split_data_list

    original_weights: dict[int, dict[str, float]] = get_ng_layer_weights(
        resolved_layer, mesh, resolved_skin_cluster
    )
    new_weights = compute_split_weights(
        mesh_shape, original_weights, resolved_split_data_collection
    )
    set_ng_layer_weights(resolved_layer, mesh, new_weights, resolved_skin_cluster)
