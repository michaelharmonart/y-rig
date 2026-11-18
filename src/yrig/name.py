from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from enum import StrEnum

LEFT_SIDE_NAME = "L"
RIGHT_SIDE_NAME = "R"
MIDDLE_SIDE_NAME = "M"
SIDE_NAMES: tuple[str, ...] = (LEFT_SIDE_NAME, RIGHT_SIDE_NAME, MIDDLE_SIDE_NAME)

GET_SIDE_REGEX = re.compile(rf"(?:(?<=_)|^)(?:{'|'.join(SIDE_NAMES)})(?=_|\d?|$)")


class Side(StrEnum):
    LEFT = "L"
    RIGHT = "R"
    MIDDLE = "M"

    @classmethod
    def from_name(cls, name: str) -> Side:
        return {
            LEFT_SIDE_NAME: cls.LEFT,
            RIGHT_SIDE_NAME: cls.RIGHT,
            MIDDLE_SIDE_NAME: cls.MIDDLE,
        }[name]


def get_side(name: str) -> Side | None:
    """
    Extracts the side token from a control name. eg. Front_Leg_L_CTL will return "L"
    Valid sides: ["L", "R", "M"]
    Args:
        name: The control name.
    Returns:
        The side token found in the name, or None if not found.
    """
    # Create a pattern that matches any of the sides preceded by "_" and followed by "_" or end of string
    match = re.search(GET_SIDE_REGEX, name)
    return Side.from_name(match.group(0)) if match else None


def filter_side(names: Iterable[str], side: Side) -> list[str]:
    """Return names matching the specified side token."""
    return [name for name in names if get_side(name) == side]


def _flip_match_side(match: re.Match[str]) -> str:
    side = match.group(0)
    if side == LEFT_SIDE_NAME:
        return RIGHT_SIDE_NAME
    if side == RIGHT_SIDE_NAME:
        return LEFT_SIDE_NAME
    return side


def flip_side(name: str) -> str:
    """
    Replaces side tokens in the name from 'L' to 'R' or vice versa,
    only when it's a distinct token (e.g., 'Front_Leg_L' becomes 'Front_Leg_R').

    Args:
        name: The original name.
    Returns:
        The renamed string.
    """
    return re.sub(GET_SIDE_REGEX, _flip_match_side, name)


def get_short_name(transform: str) -> str:
    """Return the leaf node name from a DAG path, stripping all parent namespaces.

    Maya DAG paths use ``|`` as a separator (e.g. ``|group1|joint1``).
    This function returns only the last component of such a path.

    Args:
        transform: A full or partial Maya DAG path string.

    Returns:
        The short (leaf) name without any leading path components.
    """
    return transform.rsplit("|", 1)[-1]


def normalize_name(name: str | None) -> str:
    """Normalize a name to be a more friendly filename or filepath.

    Steps: unicode normalize → encode ASCII → lowercase → spaces to underscores
    → strip non-alphanumeric characters.
    """
    if not name:
        return ""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized_name = ascii_name.strip().lower().replace(" ", "_")
    normalized_name = re.sub(r"[^a-z0-9_]", "", normalized_name)
    return normalized_name


NATURAL_SORT_REGEX = re.compile(r"(\d+)")


def natural_sort_key(value: str) -> tuple[int | str, ...]:
    """Natural string sort key.

    Examples:
        joint1 < joint2 < joint10
        apple < banana < carrot
    """
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in NATURAL_SORT_REGEX.split(value)
        if part
    )


def format_item_count(count: int, singular: str, plural: str | None = None) -> str:
    """Format a count with the appropriate singular or plural noun."""
    if plural is None:
        plural = f"{singular}s"

    return f"{count} {singular if count == 1 else plural}"
