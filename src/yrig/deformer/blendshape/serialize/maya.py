from collections.abc import Iterable
from pathlib import Path

from maya import cmds

from yrig.deformer.blendshape.target import resolve_target_index
from yrig.select import maintain_selection


def import_maya_shape_file(
    filepath: Path,
    blendshape: str | None = None,
) -> list[str]:
    """
    Create and load a blendShape from a .shape/.shp file.
    Args:
        filepath: Path to the exported shape file.
        blendshape_name: Optionally specifiy the name of the blendShape node to import to.
    Returns:
        The blendShape node(s) imported to.
    """

    if not filepath.exists():
        raise FileNotFoundError(f"Shape file does not exist: {filepath}")

    # Import into an existing blendShape.
    if blendshape and cmds.objExists(blendshape):
        cmds.blendShape(
            blendshape,
            edit=True,
            ip=str(filepath),
        )
        return [blendshape]

    # Let Maya create the blendShape from the file.
    kwargs = {}
    if blendshape:
        kwargs["name"] = blendshape

    with maintain_selection():
        cmds.select(clear=True)
        return cmds.blendShape(  # type: ignore
            **kwargs,  # type: ignore
            ip=str(filepath),
        )


def export_maya_shape_file(
    filepath: Path,
    blendshape: str,
    targets: Iterable[str | int] | None = None,
) -> None:
    """
    Export a blendShape node to a .shape/.shp file.
    Args:
        blendshape_node: Name of the blendShape node.
        filepath: Output file path.
        targets: The names or indices of the targets to export,
    """

    if not cmds.objExists(blendshape):
        raise RuntimeError(f"BlendShape does not exist: {blendshape}")

    if cmds.nodeType(blendshape) != "blendShape":
        raise TypeError(f"{blendshape} is not a blendShape node")

    # Ensure output directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if targets:
        target_indices = [resolve_target_index(blendshape, target) for target in targets]
        kwargs["exportTarget"] = [(0, index) for index in target_indices]
    # Export shape file
    cmds.blendShape(
        blendshape,
        **kwargs,  # type: ignore
        edit=True,
        export=str(filepath),
    )
