import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TypeAlias

from maya import cmds, mel

from yrig.io.core import confirm_overwrite
from yrig.select import maintain_selection

log = logging.getLogger(__name__)

FBXOption: TypeAlias = bool | int | float | str

# FBX commands that do not use the standard -v flag.
_DIRECT_OPTIONS = {
    "FBXExportUpAxis",
    "FBXExportScaleFactor",
    "FBXImportScaleFactor",
    "FBXImportScaleFactorEnable",
    "FBXImportForcedFileAxis",
}


def _load_fbx_plugin() -> None:
    """Ensure the Maya FBX plugin is loaded."""
    if not cmds.pluginInfo("fbxmaya", query=True, loaded=True):
        cmds.loadPlugin("fbxmaya")


def _mel_value(value: FBXOption) -> str:
    """Convert a Python value into a MEL argument."""
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    raise TypeError(f"Unsupported FBX option value: {value!r}")


def _set_fbx_options(
    prefix: str,
    options: Mapping[str, FBXOption],
) -> None:
    """Apply FBX plugin options through MEL.

    Option names can be supplied with or without their FBX prefix.
    For example, 'Skins' and 'FBXExportSkins' are equivalent when
    prefix is 'FBXExport'.
    """
    for name, value in options.items():
        command = name if name.startswith(prefix) else f"{prefix}{name}"

        if not re.fullmatch(r"FBX(?:Export|Import)[A-Za-z0-9_]+", command):
            raise ValueError(f"Invalid FBX option: {name}")

        if not command.startswith(prefix):
            raise ValueError(f"Invalid {prefix} option: {name}")

        flag = "" if command in _DIRECT_OPTIONS else " -v"
        mel.eval(f"{command}{flag} {_mel_value(value)};")


def import_fbx_file(
    filepath: Path,
    *,
    namespace: str | None = None,
    import_mode: str = "add",
    skins: bool = True,
    shapes: bool = True,
    cameras: bool = False,
    lights: bool = False,
    constraints: bool = False,
    fill_timeline: bool = False,
    axis_conversion: bool = True,
    merge_animation_layers: bool = True,
    protect_driven_keys: bool = True,
    generate_log: bool = False,
    reset_settings: bool = True,
    options: Mapping[str, FBXOption] | None = None,
) -> list[str]:
    """Import an FBX file and return the newly created nodes.

    Args:
        filepath: Path to the FBX file.
        namespace: Optional namespace for imported nodes.
        import_mode: 'add', 'merge', or 'exmerge'.
        skins: Import skin deformation.
        shapes: Import blendshape deformation.
        cameras: Import cameras.
        lights: Import lights.
        constraints: Import supported constraints.
        fill_timeline: Extend Maya's timeline from imported animation.
        axis_conversion: Convert the FBX coordinate system to Maya.
        merge_animation_layers: Bake imported animation layers.
        protect_driven_keys: Preserve existing driven-key channels.
        generate_log: Generate an FBX import log.
        reset_settings: Reset FBX import options before applying settings.
        options: Additional FBXImport MEL options.

    Returns:
        Names of nodes created by the import.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"No FBX file was found at {filepath}")

    if filepath.is_dir():
        raise IsADirectoryError(f"Found directory instead of file at {filepath}")

    if filepath.suffix.lower() != ".fbx":
        raise ValueError(f"Expected an FBX file: {filepath}")

    if import_mode not in {"add", "merge", "exmerge"}:
        raise ValueError(f"Invalid FBX import mode: {import_mode}")

    _load_fbx_plugin()

    if reset_settings:
        mel.eval("FBXResetImport;")

    settings: dict[str, FBXOption] = {
        "Mode": import_mode,
        "Skins": skins,
        "Shapes": shapes,
        "Cameras": cameras,
        "Lights": lights,
        "Constraints": constraints,
        "FillTimeline": fill_timeline,
        "AxisConversionEnable": axis_conversion,
        "MergeAnimationLayers": merge_animation_layers,
        "ProtectDrivenKeys": protect_driven_keys,
        "GenerateLog": generate_log,
    }

    if options:
        settings.update(options)

    _set_fbx_options("FBXImport", settings)

    kwargs = {}
    if namespace is not None:
        kwargs["namespace"] = namespace

    try:
        imported_nodes: list[str] = cmds.file(  # type: ignore
            str(filepath),
            i=True,
            type="FBX",
            ignoreVersion=True,
            returnNewNodes=True,
            **kwargs,  # type:ignore
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to import the FBX file at {filepath}") from exc

    log.info(f"Imported FBX file from {filepath}")
    return imported_nodes


def export_fbx_file(
    filepath: Path,
    nodes: Iterable[str] | None = None,
    *,
    force: bool = False,
    skins: bool = True,
    shapes: bool = True,
    smoothing_groups: bool = True,
    smooth_mesh: bool = False,
    triangulate: bool = False,
    tangents: bool = False,
    hard_edges: bool = False,
    animation: bool = True,
    bake_animation: bool = False,
    bake_start: int | None = None,
    bake_end: int | None = None,
    bake_step: int = 1,
    constraints: bool = False,
    cameras: bool = False,
    lights: bool = False,
    embedded_textures: bool = False,
    input_connections: bool = False,
    include_children: bool = True,
    instances: bool = False,
    up_axis: str = "y",
    scale_factor: float = 1.0,
    ascii: bool = False,
    file_version: str | None = None,
    generate_log: bool = False,
    reset_settings: bool = True,
    options: Mapping[str, FBXOption] | None = None,
) -> bool:
    """Export a Maya scene or collection of nodes to an FBX file.

    Args:
        filepath: Destination FBX path.
        nodes: Nodes to export. If None, export the entire scene.
        force: Overwrite existing files without prompting.
        skins: Include skin deformation.
        shapes: Include blendshape deformation.
        smoothing_groups: Export smoothing groups.
        smooth_mesh: Export subdivision information.
        triangulate: Triangulate exported geometry.
        tangents: Export tangents and binormals.
        hard_edges: Split per-vertex normals.
        animation: Include animation data.
        bake_animation: Bake animation during export.
        bake_start: First frame to bake. Defaults to playback start.
        bake_end: Last frame to bake. Defaults to playback end.
        bake_step: Frame interval when baking.
        constraints: Include supported constraints.
        cameras: Include cameras.
        lights: Include lights.
        embedded_textures: Embed texture media.
        input_connections: Include connected dependency nodes.
        include_children: Include descendants of exported nodes.
        instances: Preserve object instances.
        up_axis: Output coordinate system, 'y' or 'z'.
        scale_factor: FBX export scaling multiplier.
        ascii: Export ASCII FBX instead of binary FBX.
        file_version: Optional FBX plugin version identifier.
        generate_log: Generate an FBX export log.
        reset_settings: Reset FBX options before applying settings.
        options: Additional FBXExport MEL options.

    Returns:
        True if the file was exported, or False if cancelled.
    """
    filepath = Path(filepath)

    if filepath.suffix.lower() != ".fbx":
        raise ValueError(f"Expected an FBX file: {filepath}")

    if up_axis.lower() not in {"y", "z"}:
        raise ValueError(f"Invalid FBX up axis: {up_axis}")

    if bake_step < 1:
        raise ValueError("Bake step must be at least 1.")

    if not confirm_overwrite(filepath, force):
        return False

    _load_fbx_plugin()

    if reset_settings:
        mel.eval("FBXResetExport;")

    settings: dict[str, FBXOption] = {
        "Skins": skins,
        "Shapes": shapes,
        "SmoothingGroups": smoothing_groups,
        "SmoothMesh": smooth_mesh,
        "Triangulate": triangulate,
        "Tangents": tangents,
        "HardEdges": hard_edges,
        "BakeComplexAnimation": bake_animation,
        "BakeComplexStep": bake_step,
        "Constraints": constraints,
        "Cameras": cameras,
        "Lights": lights,
        "EmbeddedTextures": embedded_textures,
        "InputConnections": input_connections,
        "IncludeChildren": include_children,
        "Instances": instances,
        "UpAxis": up_axis.lower(),
        "ScaleFactor": scale_factor,
        "InAscii": ascii,
        "GenerateLog": generate_log,
    }

    # AnimationOnly restricts the export to animation data.
    # The standard FBX exporter includes available animation by
    # default, so disabling animation requires an additional option.
    if not animation:
        settings["AnimationOnly"] = False

    if bake_start is not None:
        settings["BakeComplexStart"] = bake_start
    elif bake_animation:
        settings["BakeComplexStart"] = int(cmds.playbackOptions(query=True, minTime=True))

    if bake_end is not None:
        settings["BakeComplexEnd"] = bake_end
    elif bake_animation:
        settings["BakeComplexEnd"] = int(cmds.playbackOptions(query=True, maxTime=True))

    if file_version is not None:
        settings["FileVersion"] = file_version

    if options:
        settings.update(options)

    _set_fbx_options("FBXExport", settings)

    # The FBX exporter accepts forward-slash paths on Windows.
    mel_path = _mel_value(filepath.resolve().as_posix())

    if nodes is not None:
        export_nodes = list(nodes)

        if not export_nodes:
            raise ValueError("No nodes were provided for FBX export.")

        with maintain_selection():
            cmds.select(*export_nodes, replace=True)
            mel.eval(f"FBXExport -f {mel_path} -s;")
    else:
        mel.eval(f"FBXExport -f {mel_path};")

    log.info(f"Exported FBX file to {filepath}")
    return True
