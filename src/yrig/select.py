from collections.abc import Generator
from contextlib import contextmanager

from maya import cmds


@contextmanager
def maintain_selection(maintain_empty: bool = False) -> Generator[None, None, None]:
    selection = cmds.ls(selection=True, long=True, ufeObjects=True, absoluteName=True) or []
    try:
        yield
    finally:
        if maintain_empty:
            cmds.select(clear=True)
        if selection:
            cmds.select(*selection, replace=True)


def get_selection() -> list[str]:
    return cmds.ls(selection=True)
