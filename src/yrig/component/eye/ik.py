from dataclasses import dataclass

from maya import cmds


@dataclass
class IKData:
    start_joint: str
    end_joint: str
    handle: str
    effector: str
    spline_curve: str
    type: str
    pole_vector: str


def create_ik_single_chain(
    name: str,
    start_joint: str,
    end_joint: str,
) -> IKData:
    ik_handle, effector = cmds.ikHandle(  # type:ignore
        name=f"{name}_IK_handle", startJoint=start_joint, endEffector=end_joint, solver="ikSCsolver"
    )

    handle = IKData(
        start_joint=start_joint,
        end_joint=end_joint,
        handle=ik_handle,
        effector=effector,
        spline_curve="",
        type="single_chain",
        pole_vector="",
    )

    return handle
