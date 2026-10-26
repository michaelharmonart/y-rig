from enum import StrEnum
from typing import TypeAlias

from maya.api.OpenMaya import (
    MFn,
    MFnMesh,
    MFnNurbsCurve,
    MFnNurbsSurface,
    MMatrix,
    MMeshIntersector,
    MPoint,
    MPointOnMesh,
    MSpace,
)

from yrig.maya_api.utils import get_dag_path, get_depend_node

MeshMirrorMap: TypeAlias = dict[int, int]
CurveMirrorMap: TypeAlias = dict[int, int]
SurfaceMirrorMap: TypeAlias = dict[tuple[int, int], tuple[int, int]]
TopologyMirrorMap: TypeAlias = MeshMirrorMap | CurveMirrorMap | SurfaceMirrorMap


class MirrorAxis(StrEnum):
    X = "x"
    Y = "y"
    Z = "z"


def get_mirror_matrix(axis: MirrorAxis | str = MirrorAxis.X) -> MMatrix:
    """Return a reflection matrix across the selected world-space plane."""
    resolved_axis = MirrorAxis(axis)
    diagonal = [1.0, 1.0, 1.0]
    diagonal[(MirrorAxis.X, MirrorAxis.Y, MirrorAxis.Z).index(resolved_axis)] = -1.0

    return MMatrix(
        (
            (diagonal[0], 0, 0, 0),
            (0, diagonal[1], 0, 0),
            (0, 0, diagonal[2], 0),
            (0, 0, 0, 1),
        )
    )


def _get_closest_vertex_index(
    mesh_fn: MFnMesh,
    point_on_mesh: MPointOnMesh,
) -> int:
    """Return the triangle vertex index nearest to the closest surface point."""
    vertices = mesh_fn.getPolygonTriangleVertices(
        point_on_mesh.face,
        point_on_mesh.triangle,
    )
    u, v = point_on_mesh.barycentricCoords
    weights: tuple[float, float, float] = (u, v, 1.0 - u - v)
    return vertices[max(range(3), key=lambda i: weights[i])]


def generate_mesh_mirror_mapping(
    mesh: str, axis: MirrorAxis | str = MirrorAxis.X, *, local: bool = True
) -> dict[int, int]:
    dag_path = get_dag_path(mesh)
    mobject = get_depend_node(mesh)
    if not dag_path.hasFn(MFn.kMesh):
        raise RuntimeError(f"{mesh} is not a valid mesh shape node")
    mfn_mesh: MFnMesh = MFnMesh(mobject)
    if local:
        matrix = MMatrix.kIdentity
    else:
        matrix: MMatrix = dag_path.inclusiveMatrix()

    mirror_matrix = get_mirror_matrix(axis)

    mesh_points = mfn_mesh.getPoints(space=MSpace.kObject if local else MSpace.kWorld)
    m_intersect: MMeshIntersector = MMeshIntersector()
    m_intersect.create(mobject, matrix)

    mirror_indices: dict[int, int] = {}
    for index, point in enumerate(mesh_points):
        mirrored_point: MPoint = point * mirror_matrix
        point_on_mesh = m_intersect.getClosestPoint(mirrored_point)
        mirrored_index = _get_closest_vertex_index(mfn_mesh, point_on_mesh)
        mirror_indices[index] = mirrored_index

    return mirror_indices


def generate_curve_mirror_mapping(
    curve: str, axis: MirrorAxis | str = MirrorAxis.X, *, local: bool = True
) -> dict[int, int]:
    dag_path = get_dag_path(curve)
    if not dag_path.hasFn(MFn.kNurbsCurve):
        raise RuntimeError(f"{curve} is not a valid NURBS curve shape")

    mfn_curve = MFnNurbsCurve(dag_path)
    space = MSpace.kObject if local else MSpace.kWorld
    mirror_matrix = get_mirror_matrix(axis)
    cv_positions = mfn_curve.cvPositions(space)

    mirror_indices: dict[int, int] = {}
    for index, point in enumerate(cv_positions):
        mirrored_point = point * mirror_matrix
        mirrored_index = min(
            range(len(cv_positions)),
            key=lambda i: (cv_positions[i] - mirrored_point).length(),
        )
        mirror_indices[index] = mirrored_index

    return mirror_indices


def generate_surface_mirror_mapping(
    surface: str, axis: MirrorAxis | str = MirrorAxis.X, *, local: bool = True
) -> dict[tuple[int, int], tuple[int, int]]:
    dag_path = get_dag_path(surface)
    if not dag_path.hasFn(MFn.kNurbsSurface):
        raise RuntimeError(f"{surface} is not a valid NURBS surface shape")

    mfn_surface = MFnNurbsSurface(dag_path)
    space = MSpace.kObject if local else MSpace.kWorld
    mirror_matrix = get_mirror_matrix(axis)
    num_u = mfn_surface.numCVsInU
    num_v = mfn_surface.numCVsInV
    cv_positions: dict[tuple[int, int], MPoint] = {
        divmod(index, num_v): point for index, point in enumerate(mfn_surface.cvPositions(space))
    }
    mirror_indices: dict[tuple[int, int], tuple[int, int]] = {}
    for (u, v), point in cv_positions.items():
        mirrored_point: MPoint = point * mirror_matrix

        mirrored_index = min(
            cv_positions,
            key=lambda index: (cv_positions[index] - mirrored_point).length(),
        )
        mirror_indices[u, v] = mirrored_index

    return mirror_indices
