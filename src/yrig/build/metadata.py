from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def git_commit(path: Path) -> str | None:
    """Return the current Git commit for the repository containing path."""
    path = path.resolve()

    if path.is_file():
        path = path.parent

    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip() or None


def yrig_commit() -> str | None:
    """Return the Git commit for the currently imported y-rig source."""
    package_root = Path(__file__).resolve().parents[2]
    return git_commit(package_root)


def write_rig_build_metadata(metadata_node: str, build_path: Path | None) -> str:
    """Create or update build provenance metadata on the rig root."""
    from maya import cmds

    if not cmds.objExists(metadata_node):
        metadata_node = cmds.createNode("network", name=metadata_node)

    attributes = {
        "yrig_commit": yrig_commit(),
        "rig_build_commit": git_commit(build_path) if build_path else None,
    }
    for attribute_name, value in attributes.items():
        if not cmds.attributeQuery(attribute_name, node=metadata_node, exists=True):
            cmds.addAttr(
                metadata_node,
                longName=attribute_name,
                dataType="string",
            )

        cmds.setAttr(
            f"{metadata_node}.{attribute_name}",
            value or "unknown",
            type="string",
        )

    log.info(f"Rig Build Metadata written to {metadata_node}: {attributes}")
    return metadata_node
