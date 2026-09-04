from maya.api.OpenMaya import MFloatArray, MObject
from maya.api.OpenMayaAnim import MFnWeightGeometryFilter

from yrig.maya_api.utils import get_dag_path, get_depend_node
from yrig.shape import get_components_of_shape, get_shape


def get_weights(deformer: str, geometry: str, components: MObject | None) -> MFloatArray:
    deformer_mobject = get_depend_node(deformer)
    weight_geo_filter: MFnWeightGeometryFilter = MFnWeightGeometryFilter(deformer_mobject)

    shape = get_shape(geometry)
    if shape is None:
        raise RuntimeError(
            f"{geometry} is not a shape node! This function expects a transform with a shape or a shape."
        )
    shape_path = get_dag_path(shape)
    resolved_components = (
        components if components is not None else get_components_of_shape(shape_path)
    )
    return weight_geo_filter.getWeights(shape_path, resolved_components)


def set_weights(
    deformer: str, geometry: str, weights: MFloatArray, components: MObject | None
) -> None:
    deformer_mobject = get_depend_node(deformer)
    weight_geo_filter: MFnWeightGeometryFilter = MFnWeightGeometryFilter(deformer_mobject)

    shape = get_shape(geometry)
    if shape is None:
        raise RuntimeError(
            f"{geometry} is not a shape node! This function expects a transform with a shape or a shape."
        )
    shape_path = get_dag_path(shape)
    resolved_components = (
        components if components is not None else get_components_of_shape(shape_path)
    )
    weight_geo_filter.setWeight(shape_path, resolved_components, weights)
