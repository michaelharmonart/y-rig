from collections.abc import Sequence

from maya import cmds


def create_proximity_wrap(
    driver: str | Sequence[str],
    driven: str | Sequence[str],
    name: str | None = None,
) -> str:
    resolved_name = name if name else f"{'_'.join(driven)}_wrap"
    wrap_node: str = cmds.deformer(driven, type="proximityWrap", name=resolved_name)[0]  # type: ignore
    cmds.proximityWrap(
        wrap_node,
        edit=True,
        addDrivers=driver,  # type: ignore
        applyUserDefaults=False,
    )

    return wrap_node
