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
