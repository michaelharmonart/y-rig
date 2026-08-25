# type:ignore
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from maya import cmds

ConstraintType = Literal[
    "parent",
    "point",
    "orient",
    "scale",
    "aim",
]


@dataclass
class Constraint:
    """Base wrapper for a Maya constraint node."""

    node: str
    driven: str
    drivers: list[str] = field(default_factory=list)

    @property
    def weights(self) -> dict[str, str]:
        """Return each driver and its corresponding weight attribute."""

        raise NotImplementedError

    @property
    def connections(self) -> list[str]:
        """Return the weight attribute plugs on the constraint."""

        return list(self.weights.values())

    def add_driver(self, driver: str, weight: float = 1.0) -> str:
        """Add another driver to the constraint."""

        raise NotImplementedError

    def remove_driver(self, driver: str) -> None:
        """Remove a driver from the constraint."""

        raise NotImplementedError

    def set_weight(self, driver: str, weight: float) -> None:
        """Set the weight associated with a driver."""

        weights = self.weights

        if driver not in weights:
            raise ValueError(f"{driver!r} is not a driver of constraint {self.node!r}.")

        cmds.setAttr(weights[driver], weight)


@dataclass
class ParentConstraint(Constraint):
    """Wrapper around a Maya parentConstraint node."""

    def refresh(self) -> None:
        """Refresh the stored driver list from the Maya constraint."""

        self.drivers = (
            cmds.parentConstraint(
                self.node,
                query=True,
                targetList=True,
            )
            or []
        )

    @property
    def weights(self) -> dict[str, str]:
        """Return driver names mapped to their weight attribute plugs."""

        drivers = (
            cmds.parentConstraint(
                self.node,
                query=True,
                targetList=True,
            )
            or []
        )

        aliases = (
            cmds.parentConstraint(
                self.node,
                query=True,
                weightAliasList=True,
            )
            or []
        )

        return {
            driver: f"{self.node}.{alias}" for driver, alias in zip(drivers, aliases, strict=False)
        }

    def add_driver(self, driver: str, weight: float = 1.0) -> str:
        """Add a new target to this parent constraint."""

        _validate_transform(driver)

        self.refresh()

        if driver in self.drivers:
            cmds.warning(f"{driver} is already a driver of {self.node}.")
            return self.weights[driver]

        cmds.parentConstraint(
            driver,
            self.node,
            edit=True,
            weight=weight,
        )

        self.refresh()
        return self.weights[driver]

    def remove_driver(self, driver: str) -> None:
        """Remove a target from this parent constraint."""

        self.refresh()

        if driver not in self.drivers:
            raise ValueError(f"{driver!r} is not a driver of {self.node!r}.")

        cmds.parentConstraint(
            driver,
            self.node,
            edit=True,
            remove=True,
        )

        self.refresh()


def constraint(
    drivers: str | Sequence[str],
    driven: str,
    constraint_type: ConstraintType = "parent",
    parent: str | None = None,
    name: str | None = None,
    maintain_offset: bool = True,
    weight: float = 1.0,
    **kwargs,
) -> Constraint:
    """Create a constraint and return its specialized wrapper.

    Args:
        drivers:
            One driver transform or a sequence of driver transforms.
        driven:
            The transform being constrained.
        constraint_type:
            The type of constraint to create.
        parent:
            Optional transform used to logically organize the constraint
            through a message connection. Maya constraint nodes cannot be
            DAG-parented.
        name:
            Optional name for the constraint node.
        maintain_offset:
            Whether to preserve the driven transform's current offset.
        weight:
            Initial weight assigned to each driver.
        **kwargs:
            Additional arguments passed to the specialized constraint creator.

    Returns:
        A specialized Constraint wrapper.
    """

    driver_list = _normalize_drivers(drivers)

    _validate_transform(driven)

    if not driver_list:
        raise ValueError("At least one driver must be provided.")

    creators = {
        "parent": _create_parent_constraint,
    }

    creator = creators.get(constraint_type)

    if creator is None:
        supported = ", ".join(sorted(creators))
        raise NotImplementedError(
            f"Constraint type {constraint_type!r} is not implemented. "
            f"Currently supported: {supported}."
        )

    result = creator(
        drivers=driver_list,
        driven=driven,
        name=name,
        maintain_offset=maintain_offset,
        weight=weight,
        **kwargs,
    )

    if parent:
        _organize_constraint(result.node, parent)

    return result


def _create_parent_constraint(
    drivers: list[str],
    driven: str,
    name: str | None = None,
    maintain_offset: bool = True,
    weight: float = 1.0,
    skip_translate: Sequence[str] | None = None,
    skip_rotate: Sequence[str] | None = None,
) -> ParentConstraint:
    """Create a parent constraint."""

    for driver in drivers:
        _validate_transform(driver)

    command_kwargs = {
        "maintainOffset": maintain_offset,
        "weight": weight,
    }

    if name:
        command_kwargs["name"] = name

    if skip_translate:
        command_kwargs["skipTranslate"] = list(skip_translate)

    if skip_rotate:
        command_kwargs["skipRotate"] = list(skip_rotate)

    node = cmds.parentConstraint(
        drivers,
        driven,
        **command_kwargs,
    )[0]

    result = ParentConstraint(
        node=node,
        driven=driven,
        drivers=list(drivers),
    )

    result.refresh()
    return result


def _normalize_drivers(
    drivers: str | Sequence[str],
) -> list[str]:
    """Normalize a single driver or sequence into a list."""

    if isinstance(drivers, str):
        return [drivers]

    return list(drivers)


def _validate_transform(node: str) -> None:
    """Ensure that a Maya transform exists."""

    if not cmds.objExists(node):
        raise ValueError(f"Maya node does not exist: {node!r}")


def _organize_constraint(
    constraint_node: str,
    parent: str,
    attribute: str = "constraintNodes",
) -> None:
    cmds.parent(constraint_node, parent)
