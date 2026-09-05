from abc import ABC, abstractmethod
from typing import ClassVar, Self

from maya import cmds

from yrig.maya_api.attribute import MessageAttribute
from yrig.maya_api.utils import ensure_plugin_loaded
from yrig.maya_api.version import MAYA_API_VERSION, TARGET_API_VERSION


def is_maya2026_or_newer() -> bool:
    return MAYA_API_VERSION >= 20260000


def is_target_2026_or_newer() -> bool:
    return TARGET_API_VERSION >= 20260000


# Mapping of Node -> Actual name depending on maya version
NODE_TYPES: dict[str, dict[str, str]] = {
    "absolute": {"standard": "absolute", "DL": "absoluteDL"},
    "multiply": {"standard": "multiply", "DL": "multiplyDL"},
    "subtract": {"standard": "subtract", "DL": "subtractDL"},
    "sum": {"standard": "sum", "DL": "sumDL"},
    "sin": {"standard": "sin", "DL": "sinDL"},
    "cos": {"standard": "cos", "DL": "cosDL"},
    "divide": {"standard": "divide", "DL": "divideDL"},
    "clampRange": {"standard": "clampRange", "DL": "clampRangeDL"},
    "distanceBetween": {"standard": "distanceBetween", "DL": "distanceBetweenDL"},
    "crossProduct": {"standard": "crossProduct", "DL": "crossProductDL"},
    "length": {"standard": "length", "DL": "lengthDL"},
    "lerp": {"standard": "lerp", "DL": "lerpDL"},
    "rowFromMatrix": {"standard": "rowFromMatrix", "DL": "rowFromMatrixDL"},
    "multiplyPointByMatrix": {
        "standard": "multiplyPointByMatrix",
        "DL": "multiplyPointByMatrixDL",
    },
    "multiplyVectorByMatrix": {
        "standard": "multiplyVectorByMatrix",
        "DL": "multiplyVectorByMatrixDL",
    },
    "normalize": {"standard": "normalize", "DL": "normalizeDL"},
}


class Node(ABC):
    """Base class for all Maya nodes."""

    node_type: ClassVar[str]
    plugin: ClassVar[str | None] = None

    def __init__(self, name: str) -> None:
        """Wrap an existing node by name. Prefer .create() or .from_existing()."""
        self.name = name
        self.message = MessageAttribute(f"{self.name}.message")
        self._setup_attributes()

    @classmethod
    def _resolve_node_type(cls) -> str:
        node_type = cls.node_type
        if node_type in NODE_TYPES:
            types = NODE_TYPES[node_type]
            if is_maya2026_or_newer() and not is_target_2026_or_newer():
                return types["DL"]
            return types["standard"]
        return node_type

    @classmethod
    def _ensure_plugin(cls) -> None:
        if cls.plugin is not None:
            ensure_plugin_loaded(cls.plugin)

    @classmethod
    def create(cls, name: str | None = None) -> Self:
        """Create a new node of this type."""
        resolved_type = cls._resolve_node_type()
        cls._ensure_plugin()
        created_name = cmds.createNode(resolved_type, name=name or cls.node_type)
        return cls(created_name)

    @classmethod
    def from_existing(cls, name: str) -> Self:
        """Wrap a node that already exists in the scene."""
        if not cmds.objExists(name):
            raise ValueError(f"Node does not exist: {name}")
        return cls(name)

    @abstractmethod
    def _setup_attributes(self) -> None:
        """Override in subclasses to define node-specific attributes."""

    def delete(self) -> None:
        """Delete this node."""
        if cmds.objExists(self.name):
            cmds.delete(self.name)

    def exists(self) -> bool:
        """Check if this node exists in Maya."""
        return cmds.objExists(self.name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __str__(self) -> str:
        return self.name
