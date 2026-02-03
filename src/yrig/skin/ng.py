import os
from functools import wraps
from typing import TYPE_CHECKING

import maya.cmds as cmds

if TYPE_CHECKING:
    from ngSkinTools2 import api as ng
    from ngSkinTools2.api.plugin import is_plugin_loaded, load_plugin
else:
    ng = None
    is_plugin_loaded = None
    load_plugin = None

HAS_NG_SKIN = False
try:
    from ngSkinTools2 import api as ng
    from ngSkinTools2.api.plugin import is_plugin_loaded, load_plugin

    HAS_NG_SKIN = True
except ImportError:
    print("ngSkinTools2 not found. Skinning sub-module features will be limited.")


def require_ng_skin(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_NG_SKIN:
            print("Error: This tool requires ngSkinTools2 to be installed.")
            return None
        if is_plugin_loaded():
            load_plugin()
        return func(*args, **kwargs)

    return wrapper


@require_ng_skin
def init_layers(shape: str) -> ng.Layers:
    skin_cluster = ng.target_info.get_related_skin_cluster(shape)
    layers = ng.layers.init_layers(skin_cluster)
    layers.add("Base Weights")
    return layers


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
def apply_ng_skin_weights(weights_file: str, geometry: str) -> None:
    """
    Applies an ngSkinTools JSON weights file to the specified geometry.
    Args:
        weights_file: The JSON weights file to read.
        geometry: The transform, shape, or skinCluster Node to apply to.
    """
    config = ng.influenceMapping.InfluenceMappingConfig()
    config.use_distance_matching = False
    config.use_name_matching = True

    if not os.path.isfile(path=weights_file):
        raise RuntimeError(f"{weights_file} doesn't exist, unable to load weights.")

    # Run the import
    ng.import_json(
        target=geometry,
        file=weights_file,
        vertex_transfer_mode=ng.transfer.VertexTransferMode.vertexId,
        influences_mapping_config=config,
    )


def write_ng_skin_weights(filepath: str, geometry: str, force: bool = False) -> None:
    """
    Writes a ngSkinTools JSON file representing the weights of the given geometry.

    Args:
        filepath: The path and filename and extension to save under.
        geometry: The transform, shape, or skinCluster Node the weights are on.
        force: If True, will automatically overwrite any existing file at the filepath specified.

    """

    # If the file exists, only write it if force = True, or after asking for confirmation.
    if os.path.isfile(path=filepath):
        if force:
            pass
        else:
            confirm: str = cmds.confirmDialog(
                title="File Overwrite",
                message=f"{filepath} already exists and will be overwritten, are you sure you want to write the file?",
                button=["Yes", "No"],
                defaultButton="Yes",
                cancelButton="No",
                dismissString="No",
            )
            if confirm == "Yes":
                pass
            else:
                return

    ng.export_json(target=geometry, file=filepath)

    return
