from collections.abc import Iterable

from maya import cmds
from maya.api.OpenMaya import (
    MDagPath,
    MDagPathArray,
    MDoubleArray,
    MFn,
    MFnComponent,
    MFnDependencyNode,
    MFnMesh,
    MFnSingleIndexedComponent,
    MIntArray,
    MObject,
    MPlug,
    MPointArray,
    MSelectionList,
    MSpace,
)
from maya.api.OpenMayaAnim import MFnSkinCluster

from yrig.maya_api.enum import (
    SkinClusterNormalizeWeights,
    SkinClusterRelativeSpaceMode,
    SkinClusterWeightDistribution,
)
from yrig.maya_api.node import SkinCluster
from yrig.maya_api.utils import get_dag_path, get_depend_node
from yrig.name import natural_sort_key
from yrig.shape import get_components_of_shape, get_shape

ZERO_TOLERANCE = 1e-7


def get_skin_clusters(geometry: str) -> list[str] | None:
    """
    Return all skinCluster deformers in a mesh's construction history.

    Queries the dependency history of the given geometry (transform or shape),
    filters for nodes of type ``skinCluster``, and returns their names.

    Args:
        geometry: The name of a geometry transform or shape node.

    Returns:
        A list of skinCluster node names if any are found, otherwise ``None``.
        The list order reflects the order returned by Maya's history query.
    """
    history = cmds.listHistory(geometry, pruneDagObjects=True) or []
    skin_clusters = cmds.ls(history, type="skinCluster")  # type: ignore
    return skin_clusters if skin_clusters else None


def get_skin_cluster(geometry: str) -> str | None:
    """
    Find the skinCluster deformer attached to a geometry.

    Walks the construction history of the given mesh and returns the first
    ``skinCluster`` node found, or ``None`` if the mesh is not skinned.

    Args:
        geometry: The name of a geometry transform or shape node.

    Returns:
        The name of the first skinCluster node in the geometry's history,
        or ``None`` if no skinCluster is present.
    """
    skin_clusters = get_skin_clusters(geometry)
    return skin_clusters[0] if skin_clusters else None


def _resolve_skin_cluster(node: str) -> SkinCluster | None:
    if cmds.nodeType(node) == "skinCluster":
        return SkinCluster.from_existing(node)
    skin_cluster = get_skin_cluster(node)
    return SkinCluster.from_existing(skin_cluster) if skin_cluster is not None else None


def get_skin_cluster_influences(skin_cluster: str | SkinCluster) -> list[str]:
    """Return the influence joints bound to a skinCluster.

    Args:
        skin_cluster: The name of the skinCluster node to query.

    Returns:
        A list of influence (joint/transform) names associated with the
        skinCluster.
    """
    return cmds.skinCluster(str(skin_cluster), query=True, influence=True)  # type: ignore


def skin_geometry(
    bind_joints: Iterable[str],
    geometry: str,
    name: str | None = None,
    *,
    dual_quaternion: bool = False,
    weight_blend: bool = False,
    support_non_rigid: bool = True,
    relative_space_mode: SkinClusterRelativeSpaceMode = SkinClusterRelativeSpaceMode.WORLD,
    normalize_weights: SkinClusterNormalizeWeights = SkinClusterNormalizeWeights.INTERACTIVE,
    weight_distribution: SkinClusterWeightDistribution = SkinClusterWeightDistribution.DISTANCE,
    max_influences: int = 6,
    maintain_max_influences: bool = False,
) -> SkinCluster:
    """
    Create a skin cluster on the given geometry using the specified bind joints.

    Args:
        bind_joints: Joint names to bind the geometry to.
        geometry: Geometry to skin.
        name: Name for the skin cluster. If None, uses the geometry name with
            a ``_SC`` suffix.
        dual_quaternion: Whether to use dual quaternion skinning. If False,
            classic linear skinning is used.
        weight_blend: Whether to use DQ weight blended mode when dual
            quaternion skinning is enabled.
        support_non_rigid: Whether to enable support for non-rigid
            transformations when dual quaternion skinning is enabled.
        relative_space_mode: Space mode used by the skin cluster.
        normalize_weights: Weight normalization mode used by the skin cluster.
        weight_distribution: Method used to distribute weights when binding.
        max_influences: Maximum number of influences allowed per vertex.
        maintain_max_influences: Whether to enforce the maximum influence count.

    Returns:
        The created skin cluster.
    """
    if not name:
        name = f"{geometry}_SC"

    shape = get_shape(geometry)

    if shape is None:
        raise RuntimeError(
            f"{geometry} is not a shape node! This function expects a transform with a shape or a shape."
        )
    if not bind_joints:
        raise ValueError("The provided bind_joints list was empty")
    mode = (2 if weight_blend else 1) if dual_quaternion else 0
    skin_cluster: str = cmds.skinCluster(  # type: ignore
        *bind_joints,
        shape,
        toSelectedBones=True,
        skinMethod=mode,
        name=name,
        normalizeWeights=normalize_weights,
        weightDistribution=weight_distribution,
        maximumInfluences=max_influences,
        obeyMaxInfluences=maintain_max_influences,
    )[0]
    skin_cluster_node = SkinCluster.from_existing(skin_cluster)
    skin_cluster_node.dqs_support_non_rigid.set(support_non_rigid)
    skin_cluster_node.relative_space_mode.set(relative_space_mode)
    return skin_cluster_node


def remove_unused_influences(geometry: str, skin_cluster: str | None = None) -> list[str]:
    """
    Removes unused joints from a skinCluster and returns the removed influences.
    Args:
        geometry: Mesh transform or shape that contains the skinCluster.
        skin_cluster: Optional explicit skinCluster node name. When
            ``None``, the first skinCluster in the shape's history is used.

    Returns:
        List of influence names that were removed from the skinCluster.
    """
    if not skin_cluster:
        skin_cluster: str | None = get_skin_cluster(geometry)
        if not skin_cluster:
            raise RuntimeError(f"No skinCluster on {geometry}")
    original_influences: set[str] = cmds.skinCluster(skin_cluster, query=True, influence=True) or []  # type: ignore
    cmds.skinCluster(skin_cluster, edit=True, removeUnusedInfluence=True)
    new_influences: set[str] = set(cmds.skinCluster(skin_cluster, query=True, influence=True) or [])  # type: ignore
    return [influence for influence in original_influences if influence not in new_influences]


def get_mesh_points(fn_mesh: MFnMesh, vertex_indices: list[int] | None = None) -> MPointArray:
    """Retrieve world-space vertex positions from a mesh function set.

    When *vertex_indices* is ``None`` every vertex position is returned.
    Otherwise only the positions at the requested indices are collected
    (in the order given).

    Args:
        fn_mesh: An ``MFnMesh`` function set already attached to the
            target mesh shape.
        vertex_indices: Optional list of specific vertex indices to
            retrieve. If ``None``, all vertices are returned.

    Returns:
        An ``MPointArray`` containing the requested vertex positions in
        world space.
    """
    mesh_points: MPointArray = MPointArray()
    if vertex_indices is None:
        mesh_points = fn_mesh.getPoints(space=MSpace.kWorld)
        vertex_indices = list(range(len(mesh_points)))
    else:
        all_points: MPointArray = fn_mesh.getPoints(space=MSpace.kWorld)
        for idx in vertex_indices:
            mesh_points.append(all_points[idx])
    return mesh_points


def get_weights_of_influence(skin_cluster: str, influence: str) -> dict[int, float]:
    """Query per-vertex skin weights for a single influence joint.

    Uses the Maya API's ``MFnSkinCluster.getPointsAffectedByInfluence``
    to efficiently retrieve only the vertices and weights associated with
    the given joint.

    Args:
        skin_cluster: The name of the skinCluster node to query.
        influence: The name of the influence whose weights are requested.

    Returns:
        A dictionary mapping vertex indices to their weight values for
        the specified joint.  Vertices with zero influence are omitted.
    """
    skin_cluster_mob: MObject = get_depend_node(skin_cluster)
    influence_dag: MDagPath = get_dag_path(influence)
    mfn_skin_cluster: MFnSkinCluster = MFnSkinCluster(skin_cluster_mob)

    components: MSelectionList
    weights: list[float]
    components, weights = mfn_skin_cluster.getPointsAffectedByInfluence(influence_dag)

    index_weights: dict[int, float] = {}
    affected_indices: list[int] = []
    for i in range(components.length()):
        _dag_path, component = components.getComponent(i)
        fn_comp: MFnSingleIndexedComponent = MFnSingleIndexedComponent(component)
        indices: list[int] = fn_comp.getElements()
        affected_indices.extend(indices)
    for index, weight in zip(affected_indices, weights, strict=True):
        index_weights[index] = weight

    return index_weights


def get_influence_index_to_name_map(skin_cluster: str) -> dict[int, str]:
    skin_cluster_mob = get_depend_node(skin_cluster)
    mfn_skin_cluster: MFnSkinCluster = MFnSkinCluster(skin_cluster_mob)
    influence_paths = mfn_skin_cluster.influenceObjects()
    influence_map: dict[int, str] = {
        mfn_skin_cluster.indexForInfluenceObject(path): MFnDependencyNode(path.node()).name()
        for path in influence_paths
    }
    return influence_map


def get_influence_name_to_index_map(skin_cluster: str) -> dict[str, int]:
    skin_cluster_mob = get_depend_node(skin_cluster)
    mfn_skin_cluster: MFnSkinCluster = MFnSkinCluster(skin_cluster_mob)
    influence_paths = mfn_skin_cluster.influenceObjects()
    influence_map: dict[str, int] = {
        MFnDependencyNode(path.node()).name(): mfn_skin_cluster.indexForInfluenceObject(path)
        for path in influence_paths
    }
    return influence_map


def get_skin_weights(
    geometry: str,
    skin_cluster: str | None = None,
    prune_zero_threshold: float | None = ZERO_TOLERANCE,
) -> dict[int, dict[str, float]]:
    """
    Retrieves skinCluster weights for all vertices of the given mesh shape.

    This function returns the non-zero skin weights per vertex, mapped to their
    associated influence (joint) names. It uses the Maya API to efficiently extract
    weights from the skinCluster deformer attached to the mesh.

    Args:
        shape (str): The name of the mesh shape node to query. Must have a skinCluster.
        skin_cluster: Optional specification of which skinCluster node.

    Returns:
        dict[int, dict[str, float]: A dictionary mapping each vertex index to a list of
        (joint_name, weight) dictionaries, including only non-zero weights.
    """
    if not skin_cluster:
        resolved_skin_cluster = get_skin_cluster(geometry)
        if not resolved_skin_cluster:
            raise RuntimeError(f"No skinCluster on {geometry}")
    else:
        resolved_skin_cluster = skin_cluster
    sel: MSelectionList = MSelectionList()
    sel.add(f"{resolved_skin_cluster}.weightList")
    weight_list_plug: MPlug = sel.getPlug(0)
    point_indices: MIntArray = weight_list_plug.getExistingArrayAttributeIndices()
    influence_map = get_influence_index_to_name_map(resolved_skin_cluster)
    weights_dict: dict[int, dict[str, float]] = {}
    for i in point_indices:
        weight_list_element_plug: MPlug = weight_list_plug.elementByLogicalIndex(i)
        weight_plug: MPlug = weight_list_element_plug.child(0)

        vert_weights: dict[str, float] = {}
        influence_indices: MIntArray = weight_plug.getExistingArrayAttributeIndices()
        for influence_index in influence_indices:
            weight_element_plug: MPlug = weight_plug.elementByLogicalIndex(influence_index)
            value: float = weight_element_plug.asDouble()
            if prune_zero_threshold is None or value > prune_zero_threshold:
                influence_name = influence_map[influence_index]
                vert_weights[influence_name] = value
        weights_dict[i] = vert_weights

    return weights_dict


def organize_weights_by_influence(
    weights: dict[int, dict[str, float]],
) -> dict[str, dict[int, float]]:
    """
    Converts dictionary of vertex weights: {vtx_index: {influence_name: weight}}
    to dictionary of influence weights: {influence: {vtx_index: weight}}
    """
    weights_by_influence: dict[str, dict[int, float]] = {}
    for vertex, influence_weights in weights.items():
        for influence, weight in influence_weights.items():
            if influence in weights_by_influence:
                weights_by_influence[influence][vertex] = weight
            else:
                weights_by_influence[influence] = {vertex: weight}
    return weights_by_influence


def get_skinned_shapes() -> dict[str, str]:
    """
    Return all shapes in the scene bound to skinClusters.

    Args:
        shapes: When True, return shape nodes instead of transforms.
        intermediate: When True, include intermediate shapes.

    Returns:
       Dictionary of skin cluster -> skinned shape.
    """
    skin_shapes: dict[str, str] = {}

    skin_clusters = cmds.ls(type="skinCluster") or []

    for skin_cluster in skin_clusters:
        geometry: list[str] = (
            cmds.skinCluster(skin_cluster, query=True, geometry=True) or []
        )  # type : ignore
        for shape in geometry:
            if cmds.getAttr(f"{shape}.intermediateObject"):
                continue
            skin_shapes[skin_cluster] = shape
    return skin_shapes


def _add_missing_influences(
    exisisting_influences: Iterable[str], needed_influences: Iterable[str], skin_cluster: str
) -> list[str]:
    influences_to_add: list[str] = sorted(
        set(needed_influences) - set(exisisting_influences), key=natural_sort_key
    )
    if influences_to_add:
        cmds.skinCluster(skin_cluster, edit=True, addInfluence=influences_to_add, weight=0.0)
    return influences_to_add


def set_skin_weights(
    shape: str,
    weights: dict[int, dict[str, float]],
    skin_cluster: str | None = None,
    normalize: bool = True,
) -> str:
    """
    Sets skinCluster weights for all vertices of the given mesh shape.

    Args:
        shape (str): The name of the mesh shape node to query. Must have a skinCluster.
        new_weights (dict): Dictionary of vertex weights: {vtx_index: {influence_name: weight}}.
        skin_cluster: Optional specification of which skinCluster node.
        normalize: When True, the given weights will additionally be normalized.

    Returns:
        str: Name of the skinCluster the weights were applied to.
    """
    if not skin_cluster:
        resolved_skin_cluster = get_skin_cluster(shape)
        if not resolved_skin_cluster:
            raise RuntimeError(f"No skinCluster on {shape}")
    else:
        resolved_skin_cluster = skin_cluster

    # Ensure all influences in new_weights exist on the skinCluster
    all_influences_in_data: set[str] = {
        influence_name for point_weights in weights.values() for influence_name in point_weights
    }
    existing_influences = set(
        cmds.skinCluster(resolved_skin_cluster, query=True, influence=True) or []  # type: ignore
    )

    _add_missing_influences(existing_influences, all_influences_in_data, resolved_skin_cluster)

    # Get the actual MFnSkinCluster to apply weights with
    shape_dag = get_dag_path(shape)
    skin_cluster_mob = get_depend_node(resolved_skin_cluster)
    sel: MSelectionList = MSelectionList()
    sel.add(f"{resolved_skin_cluster}.matrix")
    matrix_list_plug: MPlug = sel.getPlug(0)
    mfn_skin_cluster: MFnSkinCluster = MFnSkinCluster(skin_cluster_mob)

    # Get influence indices
    logical_to_physical: dict[int, int] = {}
    for i in range(matrix_list_plug.numElements()):
        logical_idx = matrix_list_plug.elementByPhysicalIndex(i).logicalIndex()
        logical_to_physical[logical_idx] = i

    influence_paths: MDagPathArray = mfn_skin_cluster.influenceObjects()
    influence_indices: dict[str, int] = {
        MFnDependencyNode(path.node()).name(): logical_to_physical[
            mfn_skin_cluster.indexForInfluenceObject(path)
        ]
        for path in influence_paths
    }

    ordered_influences: list[tuple[str, int]] = list(influence_indices.items())
    ordered_influence_names = [name for name, index in ordered_influences]
    ordered_indices_only = [index for name, index in ordered_influences]
    num_influences: int = len(ordered_influence_names)

    influence_indices_array: MIntArray = MIntArray()
    for index in ordered_indices_only:
        influence_indices_array.append(index)

    components = get_components_of_shape(shape_dag)
    component_fn: MFnComponent = MFnComponent(components)
    num_components: int = component_fn.elementCount
    # Allocate list for weights
    weights_flat: list[float] = [0.0] * (num_components * num_influences)

    # Fill weights list from new_weights dict
    for point_id, point_weights in weights.items():
        base_index = point_id * num_influences
        for influence_name, weight in point_weights.items():
            influence_index = influence_indices[influence_name]
            weights_flat[base_index + influence_index] = weight

    weights_array = MDoubleArray(weights_flat)

    if not mfn_skin_cluster.object().hasFn(MFn.kSkinClusterFilter):
        raise RuntimeError(f"Selected node {skin_cluster} is not a skinCluster")

    # Set weights
    mfn_skin_cluster.setWeights(
        shape_dag,
        components,
        influence_indices_array,
        weights_array,
        normalize=normalize,
        returnOldWeights=False,
    )
    return resolved_skin_cluster
