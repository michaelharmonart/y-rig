from collections.abc import Iterable
from typing import Self

from maya import cmds

from yrig.maya_api.attribute import (
    ArrayAttribute,
    BooleanAttribute,
    IntegerAttribute,
    MessageAttribute,
)

from .data import WeightSplitData


class WeightSplitTag:
    def __init__(self, node: str) -> None:
        self.source_influence = MessageAttribute(f"{node}.source_influence")
        self.degree = IntegerAttribute(f"{node}.degree")
        self.periodic = BooleanAttribute(f"{node}.periodic")
        self.split_influences = ArrayAttribute(f"{node}.split_influences", MessageAttribute)

    @classmethod
    def create(cls, name: str | None) -> Self:
        tag_node = cmds.createNode("network", name=name if name is not None else "weight_split_tag")
        cmds.addAttr(
            tag_node,
            longName="source_influence",
            attributeType="message",
        )
        cmds.addAttr(
            tag_node,
            longName="degree",
            attributeType="long",
        )
        cmds.addAttr(
            tag_node,
            longName="periodic",
            attributeType="bool",
        )
        cmds.addAttr(tag_node, longName="split_influences", attributeType="message", multi=True)

        return cls(tag_node)

    @classmethod
    def from_node(cls, node: str) -> Self | None:
        if not cmds.objExists(node):
            return None
        return cls(node)

    def get_weight_split_data(self) -> WeightSplitData:
        destinations = self.source_influence.connected_nodes(source=False, destination=True)
        if not destinations:
            raise RuntimeError(
                f"{self.source_influence} doesn't have a connection to an influence, maybe it was disconnected at some point?"
            )

        degree = self.degree.value
        periodic = self.periodic.value
        split_influences = [
            split_influence_attr.source_node
            for split_influence_attr in self.split_influences
            if split_influence_attr.source_node is not None
        ]

        return WeightSplitData(
            source_influence=destinations[0],
            split_influences=split_influences,
            degree=degree,
            periodic=periodic,
        )


def tag_for_weight_split(
    influence: str, split_influences: Iterable[str], degree: int = 2, periodic: bool = False
) -> WeightSplitTag:
    """Create a tag connected to an influence joint with metadata attributes describing how its weights should be split.
    This data can later be read back with `get_weight_split_data` to drive an automated weight-split operation.

    Args:
        influence: The influence joint node that will be tagged.
        split_influences: The joint/transform names that the influence's weights should be redistributed across.
        degree: Degree of the spline used for spatial weight interpolation. Defaults to 2.
        periodic: If ``True``, the generated spline curve will be periodic. Defaults to ``False``.
    """
    cmds.addAttr(
        influence,
        longName="weight_split_tag",
        attributeType="message",
    )
    tag_node = WeightSplitTag.create(name=f"{influence}_weight_split_tag")
    tag_node.source_influence.connect_to(f"{influence}.weight_split_tag")
    tag_node.degree.set(degree)
    tag_node.periodic.set(periodic)
    for i, split_influence in enumerate(split_influences):
        tag_node.split_influences[i].connect_from(f"{split_influence}.message")
    return tag_node


def get_weight_split_tag(influence: str) -> WeightSplitTag | None:
    """Retrieve the `WeightSplitTag` associated with an influence, if any.

    Checks whether the influence has a ``weight_split_tag`` message attribute
    and follows the connection back to the network node that stores the split
    metadata.

    Args:
        influence: The name of the influence joint to inspect.

    Returns:
        A `WeightSplitTag` instance wrapping the connected network
        node, or ``None`` if the influence has no weight-split tag.
    """
    if not cmds.objExists(f"{influence}.weight_split_tag"):
        return None
    sources = cmds.listConnections(f"{influence}.weight_split_tag", source=True, destination=False)
    if not sources:
        return None
    source = sources[0]
    return WeightSplitTag.from_node(source)


def get_weight_split_data_from_influences(influences: Iterable[str]) -> list[WeightSplitData]:
    weight_split_data_list = []
    for influence in influences:
        weight_split_tag = get_weight_split_tag(influence)
        if weight_split_tag is None:
            continue
        weight_split_data = weight_split_tag.get_weight_split_data()
        weight_split_data_list.append(weight_split_data)
    return weight_split_data_list
