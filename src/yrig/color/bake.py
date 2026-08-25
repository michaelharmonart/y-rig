import logging

import maya.api.OpenMaya as om2
from maya import cmds
from maya.api.OpenMaya import MColor, MFnMesh, MSelectionList

from yrig.color.convert import linear_srgb_to_rec2020, srgb_to_linear_color

log = logging.getLogger(__name__)


def get_texture_from_shader(shader: str) -> str | None:
    """
    Returns the texture node connected to the color input of a shader node if there is one connected.
    """
    shader_color_attr_map: dict[str, str] = {
        "standardSurface": "baseColor",
        "aiStandardSurface": "baseColor",
        "phong": "color",
        "surfaceShader": "outColor",
        "usdPreviewSurface": "diffuseColor",
        "openPBRSurface": "baseColor",
    }
    shader_type: str = cmds.nodeType(shader)  # type: ignore
    if shader_type not in shader_color_attr_map:
        log.warning(
            f"{shader} is a {shader_type} which isn't a recognized shader type for {get_texture_from_shader.__name__}."
            f"Supported shaders: {list(shader_color_attr_map.keys())}"
        )
        return None
    color_inputs = cmds.listConnections(
        f"{shader}.{shader_color_attr_map[shader_type]}",
        source=True,
        destination=False,
    )

    if color_inputs:
        return color_inputs[0]

    return None


def get_texture_from_mesh(shape_node: str) -> str:
    """Traverses from a shape node to find its connected texture node."""
    # Get shading groups connected to the shape
    shading_groups = cmds.listConnections(shape_node, type="shadingEngine") or []
    if not shading_groups:
        raise RuntimeError(f"No shading group on {shape_node}")

    # Get the surface shader plugged into the shading group
    shader_nodes = (
        cmds.listConnections(
            f"{shading_groups[0]}.surfaceShader", source=True, destination=False, plugs=False
        )
        or []
    )
    if not shader_nodes:
        raise RuntimeError(f"No shader assigned to {shading_groups[0]}")

    shader_node = shader_nodes[0]
    texture_node = get_texture_from_shader(shader_node)  # Your previous function
    if not texture_node:
        raise RuntimeError(
            f"No texture node found connected to the color input for shader {shader_node}"
        )

    return texture_node


def sample_from_texture(
    texture_node: str, uv_list: list[tuple[float, float]]
) -> list[tuple[float, float, float]]:
    """
    Samples a Maya texture node's color at the given uv positions

    Args:
        texture_node (str): Name of the texture node (e.g. "file1").
        uv_list (list): List of (u, v) tuples in [0,1].

    Returns:
        list of (r, g, b) float tuples.
    """

    u_list: list[float] = []
    v_list: list[float] = []

    for u, v in uv_list:
        u_list.append(u)
        v_list.append(v)

    flat_color_list: list[float] = cmds.colorAtPoint(  # type: ignore
        texture_node, coordU=u_list, coordV=v_list, output="RGB"
    )
    rgb_tuples: list[tuple[float, float, float]] = [
        (flat_color_list[i], flat_color_list[i + 1], flat_color_list[i + 2])
        for i in range(0, len(flat_color_list), 3)
    ]
    return rgb_tuples


def face_color_from_texture(mesh: str, anti_alias: bool = True) -> None:
    """
    Samples texture color at each face of the given mesh and assigns the result as per-face vertex color.

    The function traces the mesh's connected shader, extracts the associated file texture,
    and samples the color at the average UV position of each face. The resulting color
    is converted from sRGB to linear and stored as a per-face color on the mesh.

    Args:
        mesh (str): The name of the mesh transform or shape node to process.
        anti_alias (bool): If True, samples color from all UVs of the face and averages them
            for anti-aliased result. If False, samples a single color at the average UV.
    """

    shapes = cmds.listRelatives(mesh, shapes=True) or []
    if not shapes:
        raise RuntimeError(f"No shape node found for {mesh}")
    shape: str = shapes[0]

    # Prepare mesh function set
    sel: MSelectionList = om2.MSelectionList()
    sel.add(shape)
    dag = sel.getDagPath(0)
    fn_mesh: MFnMesh = om2.MFnMesh(dag)

    # Confirm that UVs are available
    uv_set_name: str = fn_mesh.currentUVSetName()
    uv_counts, uv_ids = fn_mesh.getAssignedUVs(uv_set_name)

    # Check if any UVs are assigned at all
    if not uv_ids or not any(uv_counts):
        raise RuntimeError(f"No UVs assigned on mesh: {mesh} in UV set: {uv_set_name}")

    # make sure the target shape can show vertex colors
    cmds.setAttr(f"{shape}.displayColors", 1)  # type: ignore
    cmds.setAttr(f"{shape}.displayColorChannel", "Diffuse", type="string")

    # Get texture
    texture_node = get_texture_from_mesh(shape)

    face_count: int = fn_mesh.numPolygons
    face_colors: list[MColor] = []
    face_indices: list[int] = []
    uv_sample_coords: list[tuple[float, float]] = []
    face_uv_indices: dict[int, list[int]] = {}
    for face_index in range(face_count):
        # Get UVs and vertices
        face_vertices = fn_mesh.getPolygonVertices(face_index)
        u: float = 0
        v: float = 0
        uv_list: list[tuple[float, float]] = []
        num_face_verts: int = 0
        for index, _vert_index in enumerate(face_vertices):
            vert_u, vert_v = fn_mesh.getPolygonUV(face_index, index)
            u += vert_u
            v += vert_v
            uv_list.append((vert_u, vert_v))
            num_face_verts += 1
        u_average = u / num_face_verts
        v_average = v / num_face_verts
        uv_average: tuple[float, float] = (u_average, v_average)

        if anti_alias:
            start_index = len(uv_sample_coords)
            uv_sample_coords.extend(uv_list)
            end_index = len(uv_sample_coords)
            face_uv_indices[face_index] = list(range(start_index, end_index))
        else:
            uv_sample_coords.append(uv_average)
            face_uv_indices[face_index] = [len(uv_sample_coords) - 1]

    sampled_colors: list[tuple[float, float, float]] = sample_from_texture(
        texture_node=texture_node, uv_list=uv_sample_coords
    )

    for face_index, uv_indices in face_uv_indices.items():
        colors = [sampled_colors[i] for i in uv_indices]
        # Average RGB
        avg_color = tuple(sum(channel) / len(colors) for channel in zip(*colors, strict=False))

        linear_color = linear_srgb_to_rec2020(srgb_to_linear_color(avg_color))

        color = MColor(linear_color)
        face_colors.append(color)
        face_indices.append(face_index)

    fn_mesh.setFaceColors(face_colors, face_indices)
