from collections.abc import Iterable

from maya import cmds


def add_to_set(node: str | Iterable[str], set_name: str, parent: str | None = None) -> None:
    if not cmds.objExists(set_name):
        cmds.sets(node, name=set_name)  # type: ignore
    else:
        cmds.sets(node, add=set_name)  # type: ignore

    if parent:
        if not cmds.objExists(parent):
            cmds.sets(name=parent)

        cmds.sets(set_name, add=parent)  # type: ignore
