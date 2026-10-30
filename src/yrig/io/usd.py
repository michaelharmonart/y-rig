from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

from maya import cmds

from yrig.io.core import confirm_overwrite
from yrig.maya_api.utils import ensure_plugin_loaded

P = ParamSpec("P")
R = TypeVar("R")


def require_maya_usd(func: Callable[P, R]) -> Callable[P, R]:
    """Ensure mayaUsdPlugin is loaded before calling the wrapped function."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        ensure_plugin_loaded("mayaUsdPlugin")
        return func(*args, **kwargs)

    return wrapper


@require_maya_usd
def import_model_usd(
    filepath: Path, prim_path: str | None = None, *, instances: bool = False
) -> list[str]:
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at {filepath}")
    imported_top_level_prims: list[str] = cmds.mayaUSDImport(  # type: ignore
        file=str(filepath),
        prim_path=prim_path,
        preferredMaterial="openPBRSurface",
        importInstances=instances,
    )
    return imported_top_level_prims


@require_maya_usd
def import_rig_data_usd(filepath: Path, prim_path: str | None = None) -> list[str]:
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at {filepath}")
    imported_top_level_prims: list[str] = cmds.mayaUSDImport(  # type: ignore
        file=str(filepath),
        primPath=prim_path,
    )
    return imported_top_level_prims


@require_maya_usd
def export_rig_data_usd(
    filepath: Path,
    export_roots: Sequence[str] | str | None = None,
    *,
    binary: bool = False,
    force: bool = False,
) -> bool:
    if not confirm_overwrite(filepath, force):
        return False
    cmds.mayaUSDExport(  # type: ignore
        file=str(filepath),
        exportRoots=export_roots,
        defaultUSDFormat="usdc" if binary else "usda",
        exportColorSets=False,
        exportMaterials=False,
        disableModelKindProcessor=True,
        upAxis="y",
    )
    return True
