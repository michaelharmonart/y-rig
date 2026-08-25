from collections.abc import Sequence

from maya import cmds

# TODO: Add a proper Node subclass with all the proximityWrap attributes, and add those settings here


def create_proximity_wrap(
    driver: str | Sequence[str],
    driven: str | Sequence[str],
    name: str,
) -> str:
    wrap_node: str = cmds.deformer(driven, type="proximityWrap", name=name)[0]  # type: ignore
    cmds.proximityWrap(
        wrap_node,
        edit=True,
        addDrivers=driver,  # type: ignore
        applyUserDefaults=False,
    )

    return wrap_node
