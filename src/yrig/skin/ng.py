import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from maya import cmds
from maya.api.OpenMaya import MFnComponent

from yrig.io import confirm_overwrite
from yrig.io.json import load_json
from yrig.maya_api.utils import get_dag_path
from yrig.shape import get_components_of_shape
from yrig.skin.core import (
    get_influence_index_to_name_map,
    get_influence_name_to_index_map,
    get_skin_cluster,
    organize_weights_by_influence,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ngSkinTools2 import api as ng
    from ngSkinTools2.api.plugin import (
        is_plugin_loaded,
        load_plugin,
    )
else:
    ng = None
    is_plugin_loaded = None
    load_plugin = None

HAS_NG_SKIN = False
try:
    from ngSkinTools2 import api as ng
    from ngSkinTools2.api.plugin import (
        is_plugin_loaded,
        load_plugin,
    )

    HAS_NG_SKIN = True
except ImportError:
    log.warning("ngSkinTools2 not found. Skinning sub-module features will be limited.")

P = ParamSpec("P")
R = TypeVar("R")


def require_ng_skin(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that guards a function requiring ngSkinTools2 dependency.

    If ``ngSkinTools2`` is not installed the wrapped function errors with a message instead of executing.
    When it *is* available but the Maya plug-in has not yet been
    loaded, the decorator loads it automatically before proceeding.

    Args:
        func: The function to wrap.

    Returns:
        A wrapper that either delegates to *func* or errors when ngSkinTools2 is unavailable.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not HAS_NG_SKIN:
            raise RuntimeError(
                f"Execution failed for {getattr(func, '__name__', repr(func))}. Dependency 'ngSkinTools2' is not available."
            )
        if is_plugin_loaded is not None and not is_plugin_loaded():
            load_plugin()
            log.info("Successfully loaded ngSkinTools2 plugin.")
        return func(*args, **kwargs)

    return wrapper


@require_ng_skin
def get_ng_layer(skin_cluster: str, layer_name: str | None) -> ng.Layer | None:
    """
    Gets an ngSkinTools2 layer with the given name on the specified shape.

    Args:
        skin_cluster(str): The name of the skinCluster node.
        layer_name (str): The name of the layer to retrieve. If None, the active layer will be returned.

    Returns:
        ngSkinTools2.api.layers.Layer: The layer object or None if it couldn't be found.
    """

    layers: ng.Layers = ng.Layers(skin_cluster)

    if layer_name is None:
        return layers.current_layer()

    # Check for existing layer
    for layer in layers.list():
        if layer.name == layer_name:
            return layer

    return None


@require_ng_skin
def get_ng_layer_used_influence_index_to_name_mapping(
    layer: ng.Layer, skin_cluster: str
) -> dict[int, str]:
    used_influences: list[int] = layer.get_used_influences()
    influence_map = get_influence_index_to_name_map(skin_cluster)
    return {influence_id: influence_map[influence_id] for influence_id in used_influences}


@require_ng_skin
def get_or_create_ng_layer(skin_cluster: str, layer_name: str) -> ng.Layer:
    """
    Gets or creates an ngSkinTools2 layer with the given name on the specified shape.

    Args:
        skin_cluster(str): The name of the skinCluster node.
        layer_name (str): The name of the layer to create or retrieve.

    Returns:
        ngSkinTools2.api.layers.Layer: The existing or newly created layer object.
    """

    layers: ng.Layers = ng.Layers(skin_cluster)

    # Check for existing layer
    for layer in layers.list():
        if layer.name == layer_name:
            return layer

    # Create and return new layer
    new_layer = layers.add(layer_name)
    return new_layer


@require_ng_skin
def apply_ng_skin_weights(weights_file: Path, geometry: str) -> None:
    """Apply an ngSkinTools2 JSON weights file to the specified geometry.

    Uses name-based influence matching (not distance-based) and vertex-ID
    transfer mode, so the topology of the target mesh must match the file.

    Args:
        weights_file: The JSON weights file to read.
        geometry: The transform, shape, or skinCluster Node to apply to.
    """
    config = ng.influenceMapping.InfluenceMappingConfig()
    config.use_distance_matching = False
    config.use_name_matching = True

    if not weights_file.exists():
        raise RuntimeError(f"{weights_file} doesn't exist, unable to load weights.")

    # Run the import
    ng.import_json(
        target=geometry,
        file=str(weights_file),
        vertex_transfer_mode=ng.transfer.VertexTransferMode.vertexId,
        influences_mapping_config=config,
    )


@require_ng_skin
def write_ng_skin_weights(filepath: Path, geometry: str, force: bool = False) -> bool:
    """
    Writes a ngSkinTools JSON file representing the weights of the given geometry.

    Args:
        filepath: The path and filename and extension to save under.
        geometry: The transform, shape, or skinCluster Node the weights are on.
        force: If True, will automatically overwrite any existing file at the filepath specified.
    """
    if not ng.get_layers_enabled(geometry):
        raise RuntimeError(f"{geometry} has not had ngSkinTools layers initialized.")
    if not confirm_overwrite(filepath, force):
        return False
    ng.export_json(target=geometry, file=str(filepath))
    log.info(f"The skin weights for {geometry} were written to {filepath}")
    return True


def get_influences_from_ng_skin_weights(
    filepath: Path,
) -> list[str]:
    """Return influence paths from an ngSkinTools2 JSON weights file.

    Args:
        filepath: Path to the weights file.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} doesn't exist, unable to load weights.")
    data = load_json(filepath, dict)
    return [influence["path"] for influence in data["influences"]]


@require_ng_skin
def cleanup_ng_data_nodes() -> None:
    """
    Removes the `ngst2SkinLayerData` nodes in the scene for publish.

    ngst2SkinLayerData nodes store the layer data for ngSkinTools, but their final result is baked
    into the skin cluster so they just bloat the rig file size if left in the scene.

    We once had a rig go from 450+ Mb to like 53 Mb just by removing these nodes.
    """
    ng_data_nodes: list[str] = cmds.ls(type="ngst2SkinLayerData")
    if ng_data_nodes:
        cmds.delete(ng_data_nodes)  # type: ignore
        log.info(
            f"Removed {len(ng_data_nodes)} ngst2SkinLayerData node(s) from the scene: {ng_data_nodes}"
        )


@require_ng_skin
def get_ng_layer_weights(
    layer: ng.Layer, mesh: str, skin_cluster: str | None = None
) -> dict[int, dict[str, float]]:
    resolved_skin_cluster = skin_cluster if skin_cluster is not None else get_skin_cluster(mesh)
    if resolved_skin_cluster is None:
        raise RuntimeError(
            f"Couldn't find a skinCluster on mesh: {mesh} which was determined by layer {layer}"
        )
    used_influences = get_ng_layer_used_influence_index_to_name_mapping(
        layer, resolved_skin_cluster
    )
    weights: dict[int, dict[str, float]] = {}
    for influence_id, influence_name in used_influences.items():
        flat_influence_weights: list[float] = layer.get_weights(influence_id)
        for vert_id, weight in enumerate(flat_influence_weights):
            if vert_id not in weights:
                weights[vert_id] = {}
            weights[vert_id][influence_name] = weight

    return weights


@require_ng_skin
def set_ng_layer_weights(
    layer: ng.Layer,
    mesh: str,
    weights: dict[int, dict[str, float]],
    skin_cluster: str | None = None,
) -> None:
    resolved_skin_cluster = skin_cluster if skin_cluster is not None else get_skin_cluster(mesh)
    if resolved_skin_cluster is None:
        raise RuntimeError(
            f"Couldn't find a skinCluster on mesh: {mesh} which was determined by layer {layer}"
        )
    components = get_components_of_shape(get_dag_path(mesh))
    mfn_component: MFnComponent = MFnComponent(components)
    number_of_components: int = mfn_component.elementCount

    weights_by_influence = organize_weights_by_influence(weights)
    influence_map = get_influence_name_to_index_map(resolved_skin_cluster)
    for influence_name, weights_dict in weights_by_influence.items():
        layer.set_weights(
            influence_map[influence_name],
            [weights_dict.get(i, 0) for i in range(number_of_components)],
        )
