from dataclasses import asdict

from maya import cmds

from yrig.io.json import dumps_json, loads_json

from .config import ANIMATION_DATA_NODE
from .models import (
    AnimationSequence,
    AnimationSettings,
    FrameRange,
)

DATA_ATTRIBUTE = "animationData"
SCHEMA_VERSION = 1


def get_storage_node() -> str:
    """Finds or creates the animation metadata node."""
    node = f":{ANIMATION_DATA_NODE}"

    if not cmds.objExists(node):
        node = cmds.createNode(
            "network",
            name=node,
            skipSelect=True,
        )

    if not cmds.attributeQuery(
        DATA_ATTRIBUTE,
        node=node,
        exists=True,
    ):
        cmds.addAttr(
            node,
            longName=DATA_ATTRIBUTE,
            dataType="string",
        )

    return node


def save_settings(settings: AnimationSettings) -> None:
    """Stores animation settings in the Maya scene."""
    node = get_storage_node()

    data = {
        "version": SCHEMA_VERSION,
        "settings": asdict(settings),
    }

    cmds.setAttr(
        f"{node}.{DATA_ATTRIBUTE}",
        dumps_json(data, pretty=True),
        type="string",
    )


def load_settings() -> AnimationSettings | None:
    """Restores previously saved animation settings."""
    node = f":{ANIMATION_DATA_NODE}"
    plug = f"{node}.{DATA_ATTRIBUTE}"

    if not cmds.objExists(plug):
        return None

    raw_data = cmds.getAttr(plug)

    if not raw_data:
        return None

    try:
        payload = loads_json(raw_data, type=dict)

        if payload.get("version") != SCHEMA_VERSION:
            raise ValueError("Unsupported animation data version.")

        data = payload["settings"]

        frame_range = FrameRange(**data.pop("frame_range"))

        sequences = [
            AnimationSequence(
                suffix=sequence["suffix"],
                frame_range=FrameRange(**sequence["frame_range"]),
                enabled=sequence.get("enabled", True),
            )
            for sequence in data.pop("sequences", [])
        ]

        return AnimationSettings(
            frame_range=frame_range,
            sequences=sequences,
            **data,
        )

    except (TypeError, ValueError, KeyError) as error:
        cmds.warning(f"Could not restore animation settings: {error}")
        return None
