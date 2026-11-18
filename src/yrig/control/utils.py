from maya import cmds

from yrig.name import get_side


def get_tagged_controls(side: str | None = None) -> list[str]:
    """
    Returns all transform nodes tagged as controllers via a connected controller node.

    Returns:
        list: A list of transform node names that are tagged as controllers.
    """
    controls = cmds.controller(query=True, allControllers=True) or []
    return [c for c in controls if get_side(c) == side] if side else controls  # type: ignore
