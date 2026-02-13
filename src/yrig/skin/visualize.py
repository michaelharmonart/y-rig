from typing import Any

import maya.api.OpenMaya as om2
import maya.cmds as cmds

from yrig.color.convert import lch_to_lab, oklab_to_linear_srgb
from yrig.skin.split import get_mesh_spline_weights, get_mesh_surface_weights


def visualize_weights_on_mesh(
    mesh_shape: str,
    weights_per_vertex: list[list[tuple[Any, float]]],
    influence_colors: dict[Any, om2.MColor],
) -> None:
    """Apply weighted vertex colors to a mesh for visual debugging of influence weights.

    For each vertex, the final color is computed by blending each influence's
    assigned color proportionally to its weight.  Colors are blended in Oklab
    space and converted to linear sRGB before being written as Maya vertex
    colors.

    Args:
        mesh_shape: The name of the mesh **shape** node (not the transform)
            that will receive vertex colors.
        weights_per_vertex: A list with one entry per vertex.  Each entry is a
            list of ``(influence_key, weight)`` tuples describing how much each
            influence contributes to that vertex.  The influence keys must
            match the keys in *influence_colors*.
        influence_colors: A mapping from influence key to an ``MColor`` in
            Oklab space that represents that influence's display color.

    Note:
        The mesh's ``displayColors`` attribute is enabled and
        ``displayColorChannel`` is set to ``"Diffuse"`` so that the vertex
        colors are immediately visible in the viewport.
    """

    # make sure the target shape can show vertex colors
    cmds.setAttr(f"{mesh_shape}.displayColors", 1)  # type: ignore
    cmds.setAttr(f"{mesh_shape}.displayColorChannel", "Diffuse", type="string")

    # get the MDagPaths
    msel: om2.MSelectionList = om2.MSelectionList()
    msel.add(mesh_shape)
    mesh_dag: om2.MDagPath = msel.getDagPath(0)

    # make the function set and get the points
    fn_mesh: om2.MFnMesh = om2.MFnMesh(mesh_dag)

    # get the points in world space
    mesh_points: om2.MPointArray = fn_mesh.getPoints(space=om2.MSpace.kWorld)

    vertex_colors: list[om2.MColor] = []
    vertex_indices: list[int] = []

    # iterate over the points and assign colors
    for i, point in enumerate(mesh_points):  # type: ignore
        point_color: om2.MColor = om2.MColor([0, 0, 0])
        weights: list[tuple[Any, float]] = weights_per_vertex[i]
        for transform, weight in weights:
            point_color += influence_colors[transform] * weight
        point_color_tuple: tuple[float, float, float] = tuple(point_color.getColor())
        point_color_rgb = oklab_to_linear_srgb(color=point_color_tuple)
        point_color = om2.MColor((point_color_rgb))
        vertex_colors.append(point_color)
        vertex_indices.append(i)
        # fn_mesh.setVertexColor(point_color, i)

    # Set all vertex colors at once
    fn_mesh.setVertexColors(vertex_colors, vertex_indices)


def visualize_split_weights(mesh: str, cv_transforms: list[str], degree: int = 2) -> None:
    """
    Visualizes spline-based weights as vertex colors on a mesh.

    A unique color is assigned to each CV transform and then for or every vertex on the mesh, 
    spline basis weights are evaluated to determine how much each CV contributes, 
    and those weights are used to blend the CV colors together.  
    The resulting per-vertex colors are applied to the mesh.

    Args:
        mesh (str): The mesh transform node to visualize on.
        cv_transforms (list[str]): A list of transform names representing the CVs of the curve.
        degree (int, optional): Degree of the spline curve. Defaults to 2.

    Returns:
        None
    """

    # get the shape node
    mesh_shape: str = cmds.listRelatives(mesh, shapes=True)[0]
    cv_positions: list[tuple[float, float, float]] = []
    cv_colors: dict[str, om2.MColor] = {}
    color_spread: float = 30
    for index, transform in enumerate(cv_transforms):
        position: tuple[float, float, float] = cmds.xform(  # type:ignore
            transform, query=True, worldSpace=True, translation=True
        )
        cv_positions.append(position)

        lab_color: om2.MColor = om2.MColor(
            lch_to_lab(color=(0.7, 0.2, (index * color_spread) % 360))
        )
        cv_colors[transform] = lab_color

    spline_weights_per_vertex: list[list[tuple[Any, float]]] = get_mesh_spline_weights(
        mesh_shape=mesh_shape, cv_transforms=cv_transforms, degree=degree
    )
    visualize_weights_on_mesh(
        mesh_shape=mesh_shape,
        weights_per_vertex=spline_weights_per_vertex,
        influence_colors=cv_colors,
    )
    return


def visualize_surface_split_weights(
    mesh: str, surface: str, num_influences: int = 9, degree: int = 2
) -> None:
    """
    Visualizes nurbs surface based weights as vertex colors on a mesh.

    The function assigns a unique color to each CV based on its hashed position. Then, for each vertex
    on the mesh, it computes the weighted color by blending CV colors using the spline-based weights.
    These vertex colors are set on the mesh and can be used to visually verify how influence weights
    fall off across the mesh.

    Args:
        mesh (str): The mesh transform node to visualize on.
        surface (str): The NURBS surface to use for getting UV values.
        num_influences (int): The number of imaginary "influences" to split weights along.
        degree (int, optional): Degree of the spline curve. Defaults to 2.

    Returns:
        None
    """
    # get the shape nodes
    mesh_shape: str = cmds.listRelatives(mesh, shapes=True)[0]
    surface_shape: str = cmds.listRelatives(surface, shapes=True)[0]

    influences = list(range(num_influences))
    influence_colors: dict[int, om2.MColor] = {}

    color_spread: float = 30
    for index, influence in enumerate(influences):
        lab_color: om2.MColor = om2.MColor(
            lch_to_lab(color=(0.7, 0.2, (index * color_spread) % 360))
        )
        influence_colors[influence] = lab_color

    surface_weights_per_vertex: list[list[tuple[Any, float]]] = get_mesh_surface_weights(
        mesh_shape=mesh_shape,
        surface_shape=surface_shape,
        influence_transforms=influences,
        degree=degree,
    )

    visualize_weights_on_mesh(
        mesh_shape=mesh_shape,
        weights_per_vertex=surface_weights_per_vertex,
        influence_colors=influence_colors,
    )
    return
