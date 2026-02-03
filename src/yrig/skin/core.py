import maya.api.OpenMaya as om2
import maya.cmds as cmds
from maya.api import OpenMayaAnim as oma


def get_skin_cluster(mesh: str) -> str | None:
    history = cmds.listHistory(mesh, pruneDagObjects=True) or []
    skin_clusters = cmds.ls(history, type="skinCluster")  # type: ignore
    return skin_clusters[0] if skin_clusters else None


def skin_mesh(
    bind_joints: list[str], geometry: str, name: str | None = None, dual_quaternion: bool = False
) -> str:
    """
    Creates a skinCluster on the given geometry using the specified bind joints.

    Args:
        bind_joints (list[str]): A list of joint names to bind the geometry to.
        geometry (str): The name of the geometry to be skinned.
        name (str | None, optional): The name to assign to the skinCluster.
            If None, a name will be auto-generated based on the geometry name.
        dual_quaternion (bool, optional): Whether to use dual quaternion skinning.
            Defaults to False (classic linear skinning).

    Returns:
        str: The name of the created skinCluster node.
    """
    if not name:
        name = f"{geometry}_SC"

    shape_list: list[str] = cmds.listRelatives(
        geometry, shapes=True, noIntermediate=True, children=True
    )
    if shape_list:
        shape = shape_list[0]
        skin_cluster = cmds.skinCluster(
            bind_joints,  # type: ignore
            shape,
            skinMethod=1 if dual_quaternion else 0,
            name=name,
        )
        if not isinstance(skin_cluster, str):
            raise RuntimeError()
    else:
        raise RuntimeError(f"{geometry} has no shape node!")

    return skin_cluster


def get_mesh_points(
    fn_mesh: om2.MFnMesh, vertex_indices: list[int] | None = None
) -> om2.MPointArray:
    mesh_points: om2.MPointArray = om2.MPointArray()
    if vertex_indices is None:
        mesh_points = fn_mesh.getPoints(space=om2.MSpace.kWorld)
        vertex_indices = list(range(len(mesh_points)))
    else:
        all_points: om2.MPointArray = fn_mesh.getPoints(space=om2.MSpace.kWorld)
        for idx in vertex_indices:
            mesh_points.append(all_points[idx])
    return mesh_points


def get_weights_of_influence(skin_cluster: str, joint: str) -> dict[int, float]:
    sel: om2.MSelectionList = om2.MSelectionList()
    sel.add(skin_cluster)
    sel.add(joint)
    skin_cluster_mob: om2.MObject = sel.getDependNode(0)
    joint_dag: om2.MDagPath = sel.getDagPath(1)
    mfn_skin_cluster: oma.MFnSkinCluster = oma.MFnSkinCluster(skin_cluster_mob)

    components: om2.MSelectionList
    weights: list[float]
    components, weights = mfn_skin_cluster.getPointsAffectedByInfluence(joint_dag)

    index_weights: dict[int, float] = {}
    affected_indices: list[int] = []
    for i in range(components.length()):
        dag_path, component = components.getComponent(i)
        fn_comp: om2.MFnSingleIndexedComponent = om2.MFnSingleIndexedComponent(component)
        indices: list[int] = fn_comp.getElements()
        affected_indices.extend(indices)
    for index, weight in zip(affected_indices, weights):
        index_weights[index] = weight

    return index_weights


def get_weights(shape: str, skin_cluster: str | None = None) -> dict[int, dict[str, float]]:
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
        skin_cluster = get_skin_cluster(shape)
        if not skin_cluster:
            raise RuntimeError(f"No skinCluster on {shape}")

    sel: om2.MSelectionList = om2.MSelectionList()
    sel.add(shape)
    sel.add(skin_cluster)
    shape_dag: om2.MDagPath = sel.getDagPath(0)
    skin_cluster_mob: om2.MObject = sel.getDependNode(1)
    mfn_skin_cluster: oma.MFnSkinCluster = oma.MFnSkinCluster(skin_cluster_mob)

    influence_paths = mfn_skin_cluster.influenceObjects()
    influence_map = {
        mfn_skin_cluster.indexForInfluenceObject(path): om2.MFnDependencyNode(path.node()).name()
        for path in influence_paths
    }

    # Create vertex component
    num_verts: int = om2.MFnMesh(shape_dag).numVertices
    fn_comp: om2.MFnSingleIndexedComponent = om2.MFnSingleIndexedComponent()
    vtx_components = fn_comp.create(om2.MFn.kMeshVertComponent)
    fn_comp.addElements(list(range(num_verts)))

    flat_weights: list[float]
    influence_count: int
    flat_weights, influence_count = mfn_skin_cluster.getWeights(shape_dag, vtx_components)

    weights_dict: dict[int, dict[str, float]] = {}
    for vtx_id in range(num_verts):
        start_index: int = vtx_id * influence_count
        vtx_weights: dict[str, float] = {}
        for i in range(influence_count):
            weight_value = flat_weights[start_index + i]
            if weight_value > 1e-6:
                influence_name = influence_map.get(i)
                if influence_name:
                    vtx_weights[influence_name] = weight_value
        if vtx_weights:
            weights_dict[vtx_id] = vtx_weights

    return weights_dict
