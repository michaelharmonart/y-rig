from collections.abc import Iterable
from pathlib import Path

from pxr import Usd

from yrig.maya_api.node.anim_curve import AnimCurveNode

from .apply import apply_scene_driven_keys_data
from .get import get_scene_driven_keys_data


def export_driven_key_layer(filepath: Path) -> None:
    driven_key_data = get_scene_driven_keys_data()
    driven_key_data.Export(str(filepath), args={"format": "usda"})


def import_driven_key_layers(filepaths: Path | Iterable[Path]) -> list[AnimCurveNode]:
    resolved_filepaths = (
        [str(filepaths)] if isinstance(filepaths, Path) else [str(p) for p in filepaths]
    )
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths[:] = resolved_filepaths
    anim_curves = apply_scene_driven_keys_data(stage)
    return anim_curves
