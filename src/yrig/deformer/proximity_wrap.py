from collections.abc import Sequence

from maya import cmds


def create_proximity_wrap(
    driver: str | Sequence[str],
    driven: str | Sequence[str],
    name: str | None = None,
    *,
    falloff_scale: int = 5,
) -> str:

    resolved_name = name or f"{driven if isinstance(driven, str) else '_'.join(driven)}_wrap"
    wrap_node: str = cmds.deformer(driven, type="proximityWrap", name=resolved_name)[0]  # type: ignore
    cmds.proximityWrap(
        wrap_node,
        edit=True,
        addDrivers=driver,  # type: ignore
        applyUserDefaults=False,
    )
    cmds.setAttr(f"{wrap_node}.falloffScale", falloff_scale)  # type: ignore
    return wrap_node
