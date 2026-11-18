import logging
from collections.abc import Iterable
from pathlib import Path

from pxr.Sdf import Layer

from yrig.io import confirm_overwrite
from yrig.maya_api.node.anim_curve import AnimCurveNode
from yrig.usd import compose_layers
from yrig.usd.diff import generate_delta_layer

from .apply import apply_scene_driven_keys_data
from .get import get_scene_driven_keys_data

log = logging.getLogger(__name__)


def export_driven_key_layer(
    filepath: Path, base_layer_paths: Iterable[Path] | None = None, *, force: bool = False
) -> bool:

    driven_key_data = get_scene_driven_keys_data()
    to_export: Layer = driven_key_data
    if base_layer_paths is not None:
        base_stage = compose_layers(base_layer_paths)
        base_layer = base_stage.Flatten()
        over_data = generate_delta_layer(base_layer, driven_key_data)
        to_export = over_data
    if to_export.empty:
        log.info(
            f"No driven key changes to export. No file written to {filepath}",
        )
        return False
    if not confirm_overwrite(filepath, force):
        return False
    to_export.Export(str(filepath), args={"format": "usda"})
    log.info(f"Driven keys data exported to {filepath}")
    return True


def import_driven_key_layers(layer_paths: Path | Iterable[Path]) -> list[AnimCurveNode]:
    stage = compose_layers(layer_paths)
    anim_curves = apply_scene_driven_keys_data(stage)
    log.info(f"Driven keys data imported from {layer_paths}")
    return anim_curves
