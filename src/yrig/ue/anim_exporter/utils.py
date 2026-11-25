import re

from maya import cmds


def get_namespaces() -> list[str]:
    namespaces = (
        cmds.namespaceInfo(
            ":",
            listOnlyNamespaces=True,
            recurse=True,
        )
        or []
    )

    return [namespace for namespace in namespaces if namespace not in {"UI", "shared"}]


def get_rig_namespaces(rigs: list[str]) -> list[str]:
    return [
        namespace
        for namespace in get_namespaces()
        if any(re.fullmatch(rf"{re.escape(rig)}\d*", namespace) for rig in rigs)
    ]
