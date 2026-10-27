from __future__ import annotations

from enum import IntEnum, StrEnum

from maya.api.OpenMayaAnim import MFnAnimCurve


class AimMatrixAxisMode(IntEnum):
    LOCK_AXIS = 0
    AIM = 1
    ALIGN = 2


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2
    NEG_X = 3
    NEG_Y = 4
    NEG_Z = 5

    @classmethod
    def from_str(cls, value: str) -> Axis:
        match value.lower():
            case "x":
                return cls.X
            case "y":
                return cls.Y
            case "z":
                return cls.Z
            case "-x":
                return cls.NEG_X
            case "-y":
                return cls.NEG_Y
            case "-z":
                return cls.NEG_Z
            case _:
                raise ValueError(f"{value} is not a valid Axis. It should be x,y,z or -x,-y,-z.")

    def to_tuple(self) -> tuple[int, int, int]:
        match self:
            case Axis.X:
                return (1, 0, 0)
            case Axis.Y:
                return (0, 1, 0)
            case Axis.Z:
                return (0, 0, 1)
            case Axis.NEG_X:
                return (-1, 0, 0)
            case Axis.NEG_Y:
                return (0, -1, 0)
            case Axis.NEG_Z:
                return (0, 0, -1)


class BlendShapeDeformationOrder(IntEnum):
    PRE_DEFORMATION = 0
    POST_DEFORMATION = 1


class BlendShapeInbetweenTargetType(IntEnum):
    ABSOLUTE = 0
    RELATIVE = 1


class BlendShapeInterpolation(IntEnum):
    LINEAR = 0
    SMOOTH = 1
    CUSTOM = 2


class BlendShapeOrigin(IntEnum):
    WORLD = 0
    LOCAL = 1
    USER = 2


class BlendShapePostDeformationOrder(IntEnum):
    NONE = 0
    TANGENT_SPACE = 1
    TRANSFORM_SPACE = 2


class ConditionOperation(IntEnum):
    EQUAL = 0
    NOT_EQUAL = 1
    GREATER_THAN = 2
    GREATER_OR_EQUAL = 3
    LESS_THAN = 4
    LESS_OR_EQUAL = 5


class RotateOrder(IntEnum):
    XYZ = 0
    YZX = 1
    ZXY = 2
    XZY = 3
    YXZ = 4
    ZYX = 5

    def __str__(self) -> str:
        return self.name


class MotionPathWorldUpType(IntEnum):
    SCENE_UP = 0
    OBJECT_UP = 1
    OBJECT_ROTATION_UP = 2
    VECTOR = 3
    NORMAL = 4


class MultiplyDivideOperation(IntEnum):
    NO_OPERATION = 0
    MULTIPLY = 1
    DIVIDE = 2
    POWER = 3


class PlusMinusAverageOperation(IntEnum):
    NO_OPERATION = 0
    SUM = 1
    SUBTRACT = 2
    AVERAGE = 3


class PoseInterpolatorInterpolation(IntEnum):
    LINEAR = 0
    GAUSSIAN = 1


class PoseInterpolatorPoseType(IntEnum):
    SWING_AND_TWIST = 0
    SWING_ONLY = 1
    TWIST_ONLY = 2


class PoseInterpolatorPoseControllerDataItemType(IntEnum):
    BOOL = 0
    CHAR = 1
    SHORT = 2
    INT = 3
    INT64 = 4
    FLOAT = 5
    DOUBLE = 6
    STRING = 7
    MOBJECT = 8


class SkinClusterBindMethod(IntEnum):
    CLOSEST_DISTANCE = 0
    CLOSEST_JOINT_IN_HIERARCHY = 1
    HEAT_MAP = 2
    GEODESIC_VOXEL = 3


class SkinClusterNormalizeWeights(IntEnum):
    NONE = 0
    INTERACTIVE = 1
    POST = 2


class SkinClusterRelativeSpaceMode(IntEnum):
    WORLD = 0
    LOCAL = 1
    CUSTOM = 2


class SkinClusterSkinningMethod(IntEnum):
    CLASSIC_LINEAR = 0
    DUAL_QUATERNION = 1
    WEIGHT_BLENDED = 2


class SkinClusterWeightDistribution(IntEnum):
    DISTANCE = 0
    NEIGHBORS = 1


class KeyframeInfinityType(StrEnum):
    CONSTANT = "constant"
    LINEAR = "linear"
    CYCLE = "cycle"
    CYCLE_RELATIVE = "cycleRelative"
    OSCILLATE = "oscillate"

    @property
    def ktype(self) -> int:
        match self:
            case self.CONSTANT:
                return MFnAnimCurve.kConstant
            case self.LINEAR:
                return MFnAnimCurve.kLinear
            case self.CYCLE:
                return MFnAnimCurve.kCycle
            case self.CYCLE_RELATIVE:
                return MFnAnimCurve.kCycleRelative
            case self.OSCILLATE:
                return MFnAnimCurve.kOscillate


class KeyframeTangentType(StrEnum):
    AUTO = "auto"
    AUTO_CUSTOM = "autocustom"
    AUTO_EASE = "autoease"
    AUTO_MIX = "automix"
    CLAMPED = "clamped"
    FAST = "fast"
    FLAT = "flat"
    LINEAR = "linear"
    PLATEAU = "plateau"
    SLOW = "slow"
    SPLINE = "spline"
    STEP = "step"
    STEP_NEXT = "stepnext"

    @property
    def ktype(self) -> int:
        match self:
            case KeyframeTangentType.AUTO:
                return MFnAnimCurve.kTangentAuto
            case KeyframeTangentType.AUTO_CUSTOM:
                return MFnAnimCurve.kTangentAutoCustom
            case KeyframeTangentType.AUTO_EASE:
                return MFnAnimCurve.kTangentAutoEase
            case KeyframeTangentType.AUTO_MIX:
                return MFnAnimCurve.kTangentAutoMix
            case KeyframeTangentType.CLAMPED:
                return MFnAnimCurve.kTangentClamped
            case KeyframeTangentType.FAST:
                return MFnAnimCurve.kTangentFast
            case KeyframeTangentType.FLAT:
                return MFnAnimCurve.kTangentFlat
            case KeyframeTangentType.LINEAR:
                return MFnAnimCurve.kTangentLinear
            case KeyframeTangentType.PLATEAU:
                return MFnAnimCurve.kTangentPlateau
            case KeyframeTangentType.SLOW:
                return MFnAnimCurve.kTangentSlow
            case KeyframeTangentType.SPLINE:
                return MFnAnimCurve.kTangentSmooth
            case KeyframeTangentType.STEP:
                return MFnAnimCurve.kTangentStep
            case KeyframeTangentType.STEP_NEXT:
                return MFnAnimCurve.kTangentStepNext


class UnsignedAxis(IntEnum):
    X = 0
    Y = 1
    Z = 2


class UvPinNormalOverride(IntEnum):
    AUTO = 0
    RAIL_CURVE = 1


class UvPinRelativeSpaceMode(IntEnum):
    WORLD = 0
    LOCAL = 1
    CUSTOM = 2
