from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from maya import cmds

from yrig.io import confirm_overwrite
from yrig.io.json import export_json, load_json

log = logging.getLogger(__name__)


@dataclass
class SetFileData:
    sets: dict[str, SetData]


@dataclass
class SetData:
    members: list[str]
    sets: list[str]


def get_set_members(set_name: str, ordered: bool = True, flatten: bool = False) -> list[str]:
    kwargs = {}
    if ordered:
        kwargs["ordered"] = True
    if flatten:
        kwargs["flatten"] = True
    return cmds.sets(set_name, query=True, **kwargs) or []  # type: ignore


def get_child_members_and_sets(set_name: str) -> tuple[list[str], list[str]]:
    raw_members: list[str] = get_set_members(set_name)
    members: list[str] = []
    sets: list[str] = []
    for member in raw_members:
        try:
            if cmds.nodeType(member) == "objectSet":
                sets.append(member)
            else:
                members.append(member)
        except RuntimeError:
            # The member is a component or other non-node.
            members.append(member)
    return members, sets


def _get_set_data(set_name: str) -> SetData:
    members, set_names = get_child_members_and_sets(set_name)
    return SetData(members=members, sets=set_names)


def _get_set_file_data(sets: Iterable[str] | str) -> SetFileData:
    resolved_sets = [sets] if isinstance(sets, str) else sets

    set_data: dict[str, SetData] = {}
    visited: set[str] = set()

    def collect(set_name: str) -> None:
        if set_name in visited:
            return

        visited.add(set_name)

        data = _get_set_data(set_name)
        set_data[set_name] = data

        for child_set in data.sets:
            collect(child_set)

    for set_name in resolved_sets:
        collect(set_name)

    return SetFileData(sets=set_data)


def export_sets(filepath: Path, sets: Iterable[str] | str, *, force: bool = False) -> bool:
    if filepath.suffix != ".ysets":
        raise ValueError("Sets files should use the .ysets extension.")

    if not confirm_overwrite(filepath, force):
        return False

    set_file_data = _get_set_file_data(sets)

    export_json(filepath, set_file_data)
    log.info(f"Exported .ysets file to {filepath}")
    return True


def _import_set_data(
    set_file_data: SetFileData, set_name: str, set_data: SetData, parent: str | None = None
) -> list[str]:
    created_sets: list[str] = []
    if not cmds.objExists(set_name):
        cmds.createNode("objectSet", name=set_name)
        created_sets.append(set_name)

    if parent is not None:
        if not cmds.objExists(parent):
            cmds.createNode("objectSet", name=parent)
            created_sets.append(parent)
        cmds.sets(set_name, addElement=parent)

    if set_data.members:
        cmds.sets(*set_data.members, addElement=set_name)
    for child_set_name in set_data.sets:
        created_child_sets = _import_set_data(
            set_file_data, child_set_name, set_file_data.sets[child_set_name], parent=set_name
        )
        created_sets.extend(created_child_sets)
    return created_sets


def import_sets(
    filepath: Path, sets: Iterable[str] | str | None = None, *, parent: str | None = None
) -> list[str]:
    set_file_data = load_json(filepath, SetFileData)
    if sets is None:
        sets_to_import = None
    else:
        sets_to_import = {sets} if isinstance(sets, str) else set(sets)

    if sets_to_import is None:
        set_name_data_pairs = set_file_data.sets.items()
    else:
        set_name_data_pairs = [
            (set_name, set_file_data.sets[set_name]) for set_name in sets_to_import
        ]

    created_sets: list[str] = []
    for set_name, set_data in set_name_data_pairs:
        created_sets.extend(_import_set_data(set_file_data, set_name, set_data, parent))
    log.info(f"Imported .ysets file from {filepath}")
    return created_sets


def add_to_set(node: str | Iterable[str], set_name: str, parent: str | None = None) -> None:
    if not cmds.objExists(set_name):
        cmds.sets(node, name=set_name)  # type: ignore
    else:
        cmds.sets(node, addElement=set_name)  # type: ignore

    if parent:
        if not cmds.objExists(parent):
            cmds.sets(name=parent)

        cmds.sets(set_name, addElement=parent)
