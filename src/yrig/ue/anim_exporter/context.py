from pathlib import Path

from maya import cmds

from . import config
from .models import FrameRange
from .utils import get_rig_namespaces


def get_timeline_frame_range() -> FrameRange:
    """Returns the visible playback range."""
    return FrameRange(
        start=int(cmds.playbackOptions(query=True, minTime=True)),
        end=int(cmds.playbackOptions(query=True, maxTime=True)),
    )


def get_pipeline_frame_range() -> FrameRange:
    """Returns the frame range defined by the pipeline."""
    # TODO: Integrate Lantern shot frame range.
    raise NotImplementedError("Pipeline frame range is not implemented yet.")


def get_default_frame_range() -> FrameRange:
    """Determines the initial frame range for the tool."""
    return get_timeline_frame_range()


def get_shot_name() -> str:
    """Resolves the current shot name."""
    scene = cmds.file(query=True, sceneName=True)

    if not isinstance(scene, str) or not scene:
        return ""

    return Path(scene).stem


def get_export_directory() -> Path:
    """Resolves the animation publish directory."""
    return config.EXPORT_DIRECTORY


def get_available_rigs() -> list[str]:
    """Returns valid rig asset names."""
    return config.AVAILABLE_RIGS


def get_available_rig_namespaces() -> list[str]:
    return get_rig_namespaces(get_available_rigs())
