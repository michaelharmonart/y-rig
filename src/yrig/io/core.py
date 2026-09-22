import logging
from collections.abc import Iterable
from pathlib import Path

from maya import cmds

log = logging.getLogger(__name__)


def confirm_overwrite(filepath: Path | Iterable[Path], force: bool = False) -> bool:
    """
    If *filepath* does not exist, return ``True`` immediately.

    If *filepath* already exists, show a confirmation dialogue and return
    ``True`` only if the user explicitly agrees to overwrite or if *force* is ``True``.
    """
    if isinstance(filepath, Path):
        resolved_paths = (filepath,)
    else:
        resolved_paths = tuple(filepath)
    existing_paths: list[Path] = []
    for path in resolved_paths:
        if path.is_dir():
            raise IsADirectoryError(f"Found directory instead of file at {filepath}")
        if path.exists():
            existing_paths.append(path)
    if force or not existing_paths:
        return True

    existing_paths_str = ", \n".join(str(path) for path in existing_paths)
    confirm: str = cmds.confirmDialog(
        title="File Overwrite",
        message="The following file(s) already exist and will be overwritten, are you sure you want to write them?\n"
        f"{existing_paths_str}",
        button=["Yes", "No"],
        defaultButton="Yes",
        cancelButton="No",
        dismissString="No",
    )
    return confirm == "Yes"


def promt_user_for_directory(message: str = "Select Directory") -> Path:
    """Prompt the user to select a directory and return it as a Path object."""
    result = cmds.fileDialog2(fileMode=3, dialogStyle=2, caption=message)
    if result and len(result) > 0:
        return Path(result[0])
    else:
        raise RuntimeError("No directory selected.")
