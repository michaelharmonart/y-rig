import re

from .models import AnimationSettings


def validate_name(name: str) -> str:
    name = name.strip()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid animation name: {name!r}")

    return name


def resolve_base_name(
    shot_name: str,
    settings: AnimationSettings,
) -> str:
    if settings.override_name:
        return validate_name(settings.custom_name)

    return validate_name(shot_name)


def resolve_animation_name(
    base_name: str,
    suffix: str = "",
) -> str:
    base_name = validate_name(base_name)

    if suffix:
        suffix = validate_name(suffix)
        return f"{base_name}_{suffix}"

    return base_name
