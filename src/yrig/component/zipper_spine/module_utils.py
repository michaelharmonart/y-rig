from dataclasses import dataclass

from maya import cmds

from yrig.control.core import Control
from yrig.transform.utils import create_space_switch, create_transform


@dataclass
class DriverOffset:
    pos: str
    offset: str


@dataclass
class Module:
    main_grp: str
    control_grp: str
    guts: str
    fk_grp: str
    ik_grp: str
    bendy_grp: str


def module_prep(
    part: str, side: str, parent: str, fkik: bool = False, gut: bool = True, bendy: bool = True
) -> Module:
    cmds.refresh()
    main_grp = create_transform(name=f"{part}_{side}", parent=parent)
    control_grp = create_transform(name=f"{part}_CTRLS_{side}", parent=main_grp)
    if gut:
        guts = create_transform(name=f"{part}_GUTS_{side}", parent=main_grp)
        cmds.hide(guts)
    else:
        guts = ""
    if fkik:
        ik_control_grp = create_transform(name=f"{part}_IK_controls_{side}", parent=control_grp)
        fk_control_grp = create_transform(name=f"{part}_Fk_controls_{side}", parent=control_grp)
    else:
        ik_control_grp = ""
        fk_control_grp = ""
    if bendy:
        bendy_control_grp = create_transform(
            name=f"{part}_bendy_controls_{side}", parent=control_grp
        )
    else:
        bendy_control_grp = ""

    prep = Module(
        main_grp=main_grp,
        control_grp=control_grp,
        guts=guts,
        fk_grp=fk_control_grp,
        ik_grp=ik_control_grp,
        bendy_grp=bendy_control_grp,
    )

    return prep


def module_space(space_list: list, control: Control) -> None:
    if len(space_list) == 0:
        pass
    elif len(space_list) == 1:
        cmds.parentConstraint(space_list[0], control.offset, maintainOffset=True)
    else:
        create_space_switch(
            target_transform=control.offset, parents=space_list, target_control=control.transform
        )


def create_module_switch(
    driven: list[str],
    drivers: dict[str, dict[str, list[str]]],
    node_attr: str,
    descriptor: str,
    controls: list[str] | None = None,
    connect_switch_attr: str | None = None,
    maintain_offset: bool = True,
) -> str | None:
    """
    Connect multiple driver systems to a shared driven chain.

    Supports:
        1 driver  -> Direct parent constraints.
        2 drivers -> Float blend switch (0-1).
        3+ drivers -> Enum switch with condition nodes.

    Args:
        driven:
            Transforms receiving constraints.

        drivers:
            Dictionary of driver systems.

            {
                "FK": {
                    "joints": [...],
                    "visibility": [...]
                },
                "IK": {
                    "joints": [...],
                    "visibility": [...]
                }
            }

        node_attr:
            Node receiving the switch attribute.

        descriptor:
            Naming prefix for generated nodes.

        controls:
            Optional controls receiving proxy switch attributes.

        connect_switch_attr:
            Optional existing switch attribute.

        maintain_offset:
            Maintain offsets when creating constraints.

    Returns:
        Switch attribute path, or None for a single driver.
    """

    if not drivers:
        raise ValueError("At least one driver system is required.")

    if not driven:
        raise ValueError("At least one driven transform is required.")

    mode_names = list(drivers)
    mode_count = len(mode_names)

    # -----------------------------------------
    # Validate driver chains
    # -----------------------------------------

    for mode_name, data in drivers.items():
        joints = data.get("joints", [])

        if len(joints) != len(driven):
            raise ValueError(
                f"{mode_name} has {len(joints)} drivers, "
                f"but {len(driven)} driven transforms were provided."
            )

    # -----------------------------------------
    # Single driver: direct constraints
    # -----------------------------------------

    if mode_count == 1:
        mode_name = mode_names[0]
        driver_joints = drivers[mode_name]["joints"]

        for driver, target in zip(
            driver_joints,
            driven,
            strict=True,
        ):
            cmds.parentConstraint(driver, target)

        return None

    # -----------------------------------------
    # Create or reuse switch attribute
    # -----------------------------------------

    if connect_switch_attr:
        switch_attr = connect_switch_attr

    else:
        if mode_count == 2:
            cmds.addAttr(
                node_attr,
                longName="module_switch",
                attributeType="double",
                defaultValue=0,
                minValue=0,
                maxValue=1,
                keyable=True,
            )

        else:
            cmds.addAttr(
                node_attr,
                longName="module_switch",
                attributeType="enum",
                enumName=":".join(mode_names),
                defaultValue=0,
                keyable=True,
            )

        switch_attr = f"{node_attr}.module_switch"

    # -----------------------------------------
    # Two drivers: float blend
    # -----------------------------------------

    """if mode_count == 2:
        reverse = ReverseNode(name=f"{descriptor}_switch_rev")

        reverse.input.x.connect_from(switch_attr)

        weights = [
            reverse.output.x,
            switch_attr,
        ]

        for index, mode_name in enumerate(mode_names):
            visibility_groups = drivers[mode_name].get("visibility", [])

            for group in visibility_groups:
                cmds.connectAttr(
                    weights[index],
                    f"{group}.visibility",
                    force=True,
                )

        # Create constraints for each driven transform

        for joint_index, target in enumerate(driven):
            driver_joints = [drivers[mode_name]["joints"][joint_index] for mode_name in mode_names]
        
            parent_con = constraint(
                drivers=driver_joints,
                driven=target,
                constraint_type="parent",
                maintain_offset=maintain_offset,
            )

            for weight_index, weight_attr in enumerate(weights):
                cmds.connectAttr(
                    weight_attr,
                    parent_con.connections[weight_index],
                    force=True,
                )

    # -----------------------------------------
    # Three or more drivers: enum switching
    # -----------------------------------------

    else:
        mode_outputs = []

        for index, mode_name in enumerate(mode_names):
            condition = cmds.createNode(
                "condition",
                name=f"{descriptor}_{mode_name}_switch_CND",
            )

            cmds.setAttr(
                f"{condition}.operation",
                0,  # Equal
            )

            cmds.setAttr(
                f"{condition}.secondTerm",
                index,
            )

            cmds.setAttr(
                f"{condition}.colorIfTrueR",
                1,
            )

            cmds.setAttr(
                f"{condition}.colorIfFalseR",
                0,
            )

            cmds.connectAttr(
                switch_attr,
                f"{condition}.firstTerm",
            )

            output = f"{condition}.outColorR"

            mode_outputs.append(output)

            # Visibility

            visibility_groups = drivers[mode_name].get("visibility", [])

            for group in visibility_groups:
                cmds.connectAttr(
                    output,
                    f"{group}.visibility",
                    force=True,
                )

        # Create constraints

        for joint_index, target in enumerate(driven):
            driver_joints = [drivers[mode_name]["joints"][joint_index] for mode_name in mode_names]

            parent_con = constraint(
                drivers=driver_joints,
                driven=target,
                constraint_type="parent",
                maintain_offset=maintain_offset,
            )

            for index, output in enumerate(mode_outputs):
                cmds.connectAttr(
                    output,
                    parent_con.connections[index],
                    force=True,
                )"""

    # -----------------------------------------
    # Proxy attributes
    # -----------------------------------------

    if controls:
        for control in controls:
            cmds.addAttr(
                control,
                longName="module_switch",
                proxy=switch_attr,
            )

    return switch_attr
