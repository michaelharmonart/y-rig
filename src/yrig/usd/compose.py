from collections.abc import Iterable
from pathlib import Path

from pxr import Usd


def compose_layers(layer_paths: Path | Iterable[Path]) -> Usd.Stage:
    if isinstance(layer_paths, Path):
        layer_paths = [layer_paths]

    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths[:] = [str(path.resolve()) for path in layer_paths]
    return stage
