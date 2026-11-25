from dataclasses import dataclass, field


@dataclass
class FrameRange:
    start: int
    end: int

    def validate(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Invalid frame range: {self.start} - {self.end}")

    def as_tuple(self) -> tuple[int, int]:
        return self.start, self.end


@dataclass
class AnimationSequence:
    suffix: str
    frame_range: FrameRange
    enabled: bool = True


@dataclass
class AnimationSettings:
    frame_range: FrameRange

    sequences: list[AnimationSequence] = field(default_factory=list)

    export_full: bool = True
    export_model: bool = False

    override_name: bool = False
    custom_name: str = ""

    namespace: str = ""
