import hashlib

from maya import cmds
from maya.api.OpenMaya import (
    MDagPath,
    MFn,
    MFnDoubleIndexedComponent,
    MFnMesh,
    MFnNurbsCurve,
    MFnNurbsSurface,
    MFnSingleIndexedComponent,
    MObject,
)

from yrig.io.json import dumps_json
from yrig.maya_api.utils import get_dag_path
from yrig.select import maintain_selection


def get_shape(object: str) -> str | None:
    """
    Return the first non-intermediate shape node associated with a DAG object.

    If the input is a transform, its child shapes are queried and the first
    valid (non-intermediate) shape is returned. If the input is already a
    shape node, it is returned directly. If no valid shape is found, ``None``
    is returned.

    Args:
        object: Name of a Maya DAG node (transform or shape).

    Returns:
        The name of the associated shape node, or ``None`` if no shape exists.
    """
    shape: str
    if cmds.nodeType(object) == "transform":
        shape_list: list[str] = cmds.listRelatives(
            object, shapes=True, noIntermediate=True, children=True
        )
        if shape_list:
            shape = shape_list[0]
            return shape
        else:
            return None

    if cmds.objectType(object, isAType="shape"):
        shape = object
        return shape
    else:
        return None


def get_components_of_shape(shape_dag_path: MDagPath) -> MObject:

    if shape_dag_path.hasFn(MFn.kMesh):
        fn = MFnMesh(shape_dag_path)
        comp_fn = MFnSingleIndexedComponent()
        component = comp_fn.create(MFn.kMeshVertComponent)
        comp_fn.addElements(range(fn.numVertices))
        return component

    if shape_dag_path.hasFn(MFn.kNurbsCurve):
        fn = MFnNurbsCurve(shape_dag_path)
        comp_fn = MFnSingleIndexedComponent()
        component = comp_fn.create(MFn.kCurveCVComponent)
        comp_fn.addElements(range(fn.numCVs))
        return component

    if shape_dag_path.hasFn(MFn.kNurbsSurface):
        fn = MFnNurbsSurface(shape_dag_path)
        comp_fn = MFnDoubleIndexedComponent()
        component = comp_fn.create(MFn.kSurfaceCVComponent)
        for u in range(fn.numCVsInU):
            for v in range(fn.numCVsInV):
                comp_fn.addElement(u, v)
        return component
    else:
        raise TypeError(f"Unsupported shape type: {shape_dag_path.node().apiTypeStr}")


def shape_topology_signature(shape: str) -> tuple:
    """Return a hashable signature describing a shape's topology."""
    shape = get_shape(shape) or shape
    dag_path = get_dag_path(shape)

    if dag_path.hasFn(MFn.kMesh):
        fn = MFnMesh(dag_path)
        polygon_counts, polygon_connects = fn.getVertices()

        return (
            "mesh",
            fn.numVertices,
            fn.numPolygons,
            tuple(polygon_counts),
            tuple(polygon_connects),
        )

    if dag_path.hasFn(MFn.kNurbsCurve):
        fn = MFnNurbsCurve(dag_path)

        return (
            "nurbsCurve",
            fn.numCVs,
            fn.degree,
            fn.form,
            fn.numSpans,
            tuple(fn.knots()),
        )

    if dag_path.hasFn(MFn.kNurbsSurface):
        fn = MFnNurbsSurface(dag_path)

        return (
            "nurbsSurface",
            fn.numCVsInU,
            fn.numCVsInV,
            fn.degreeU,
            fn.degreeV,
            fn.formInU,
            fn.formInV,
            fn.numSpansInU,
            fn.numSpansInV,
            tuple(fn.knotsInU()),
            tuple(fn.knotsInV()),
        )

    raise TypeError(f"Unsupported shape type: {dag_path.node().apiTypeStr}")


def shape_topology_hash(shape: str) -> str:
    """Return a SHA-256 hash of a shape's topology."""
    signature = shape_topology_signature(shape)
    signature_json = dumps_json(signature, pretty=False)
    return hashlib.sha256(signature_json.encode()).hexdigest()


def bake_shape(transform: str, zero_pivot: bool = True) -> None:
    cmds.makeIdentity(transform, apply=True)
    if zero_pivot:
        cmds.xform(transform, pivots=(0, 0, 0))


def set_smooth_preview(geometry: str, enable: bool = True) -> None:
    with maintain_selection(maintain_empty=True):
        cmds.select(geometry, replace=True)
        if enable:
            cmds.displaySmoothness(polygonObject=3)
        else:
            cmds.displaySmoothness(polygonObject=0)
