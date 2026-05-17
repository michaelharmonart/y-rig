import logging
import tempfile
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

import maya.cmds as cmds
import msgspec

from yrig.name import get_short_name, normalize_name
from yrig.util import confirm_overwrite

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
def init_layers(shape: str) -> ng.Layers:
    """Initialize ngSkinTools2 layers on a mesh shape and add a default base layer.

    Looks up the skinCluster associated with *shape*, initialises the
    ngSkinTools2 layer stack on it, and creates an initial ``"Base Weights"``
    layer that acts as the foundation for subsequent paint layers.

    Args:
        shape: The mesh shape node (not the transform) that has a
            skinCluster attached.

    Returns:
        The ``ngSkinTools2.api.Layers`` object managing the layer stack
        for the given shape's skinCluster.
    """
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


def _load_ng_skin_data(filepath: Path) -> dict:
    if filepath.name == "manifest.json":
        return _combine_ng_skin_file_layers_data(filepath)
    with open(filepath, "rb") as f:
        return msgspec.json.decode(f.read())


def get_influences_from_ng_skin_weights(
    filepath: Path,
) -> list[str]:
    """Return influence paths from an ngSkinTools2 JSON weights file.

    Args:
        filepath: Path to the weights file.
    """
    if not filepath.exists():
        raise RuntimeError(f"{filepath} doesn't exist, unable to load data.")
    data = _load_ng_skin_data(filepath)
    return [influence["path"] for influence in data["influences"]]


def _run_ng_import(filepath: Path, geometry: str) -> None:
    config = ng.influenceMapping.InfluenceMappingConfig()
    config.use_distance_matching = False
    config.use_name_matching = True
    ng.import_json(
        target=geometry,
        file=str(filepath),
        vertex_transfer_mode=ng.transfer.VertexTransferMode.vertexId,
        influences_mapping_config=config,
    )


def _apply_ng_skin_from_manifest(manifest: Path, geometry: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json") as file:
        temp_path = Path(file.name)
        combine_ng_skin_file_by_layers(manifest, temp_path)
        _run_ng_import(temp_path, geometry)


@require_ng_skin
def apply_ng_skin_weights(weights_file: Path, geometry: str) -> None:
    """Apply an ngSkinTools2 JSON weights file to the specified geometry.

    Uses name-based influence matching (not distance-based) and vertex-ID
    transfer mode, so the topology of the target mesh must match the file.

    Args:
        weights_file: The JSON weights file to read (either a single file, or the manifest for multi-file).
        geometry: The transform, shape, or skinCluster Node to apply to.
    """
    config = ng.influenceMapping.InfluenceMappingConfig()
    config.use_distance_matching = False
    config.use_name_matching = True

    if not weights_file.exists():
        raise RuntimeError(f"{weights_file} doesn't exist, unable to load weights.")

    if weights_file.name == "manifest.json":
        _apply_ng_skin_from_manifest(weights_file, geometry)
    else:
        _run_ng_import(weights_file, geometry)


@require_ng_skin
def write_ng_skin_weights(filepath: Path, geometry: str, force: bool = False) -> None:
    """
    Writes a ngSkinTools JSON file representing the weights of the given geometry.

    Args:
        filepath: The path and filename and extension to save under.
        geometry: The transform, shape, or skinCluster Node the weights are on.
        force: If True, will automatically overwrite any existing file at the filepath specified.

    """
    if not confirm_overwrite(filepath):
        return
    ng.export_json(target=geometry, file=str(filepath))


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


def split_ng_skin_file_by_layers(
    skin_file: Path, output_path: Path, layers_to_write: set[str] | None = None
) -> None:
    with open(skin_file, mode="rb") as file:
        data: dict = msgspec.json.decode(file.read())
    mesh: dict[str, list] = data["mesh"]
    influences: list[dict] = data["influences"]
    layers: list[dict] = data["layers"]

    manifest_data: dict = {"manifest": True, "layers": []}
    layer_file_map: dict[Path, dict] = {}
    layer_id_map: dict[int, dict] = {}
    for layer in layers:
        layer_id: int = layer["id"]
        layer_id_map[layer_id] = layer
        layer_name: str = layer["name"]

        if layer["parentId"] is None:
            layer_filepath = Path(f"{normalize_name(layer_name)}.nglayer")
            manifest_data["layers"].append(
                {"id": layer_id, "name": layer_name, "path": layer_filepath.as_posix()}
            )
            if layers_to_write is None:
                layer_file_map[layer_filepath] = layer
            elif layer["name"] in layers_to_write:
                layer_file_map[layer_filepath] = layer

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "manifest.json", mode="wb") as file:
        encoded = msgspec.json.encode(manifest_data)
        formatted = msgspec.json.format(encoded)
        file.write(formatted)

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "mesh_data.json", mode="wb") as file:
        encoded = msgspec.json.encode({"mesh": mesh})
        file.write(encoded)

    for filepath, layer in layer_file_map.items():
        file_layers = [layer]
        layer_children: list[int] = layer["children"]
        file_layers.extend(layer_id_map[layer_id] for layer_id in layer_children)
        with open(output_path / filepath, mode="wb") as file:
            encoded = msgspec.json.encode({"influences": influences, "layers": file_layers})
            file.write(encoded)


def _combine_ng_skin_file_layers_data(manifest: Path) -> dict:
    with open(manifest, mode="rb") as file:
        manifest_data: dict = msgspec.json.decode(file.read())

    with open(manifest.parent / "mesh_data.json", mode="rb") as file:
        mesh_data: dict = msgspec.json.decode(file.read())

    merged_influences: list[dict] = []
    merged_layers: list[dict] = []

    influence_name_to_index: dict[str, int] = {}  # name -> new canonical ID
    next_influence_id: int = 0

    for layer_entry in manifest_data["layers"]:
        layer_filepath = Path(layer_entry["path"])
        with open(manifest.parent / layer_filepath, mode="rb") as file:
            layer_data: dict = msgspec.json.decode(file.read())

        local_influences: list[dict] = layer_data["influences"]
        local_layers: list[dict] = layer_data["layers"]

        # Build a remap from this file's influence index -> merged indices
        local_index_to_merged: dict[int, int] = {}
        for influence in local_influences:
            name: str = get_short_name(influence["path"])
            local_index: int = influence["index"]
            if name not in influence_name_to_index:
                influence_name_to_index[name] = next_influence_id
                new_influence = influence.copy()
                new_influence["index"] = next_influence_id
                merged_influences.append(new_influence)
                next_influence_id += 1
            local_index_to_merged[local_index] = influence_name_to_index[name]

        for layer in local_layers:
            layer["influences"] = {
                str(local_index_to_merged[int(k)]): v for k, v in layer["influences"].items()
            }
            merged_layers.append(layer)

    combined: dict = {
        "mesh": mesh_data["mesh"],
        "influences": merged_influences,
        "layers": merged_layers,
    }

    return combined


def combine_ng_skin_file_by_layers(manifest: Path, output_file: Path) -> None:
    combined = _combine_ng_skin_file_layers_data(manifest)
    with open(output_file, mode="wb") as file:
        encoded = msgspec.json.encode(combined)
        file.write(encoded)
