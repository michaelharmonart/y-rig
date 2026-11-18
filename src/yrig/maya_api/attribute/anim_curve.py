from typing import Generic, TypeVar

from .core import Attribute, ScalarAttribute

InputAttribute = TypeVar("InputAttribute", bound=ScalarAttribute)
OutputAttribute = TypeVar("OutputAttribute", bound=ScalarAttribute)


class AnimCurveKeyTimeValueAttribute(
    Attribute,
    Generic[InputAttribute, OutputAttribute],
):
    def __init__(
        self,
        attr_path: str,
        input_type: type[InputAttribute],
        output_type: type[OutputAttribute],
    ) -> None:
        super().__init__(attr_path)

        self.key_time = input_type(f"{attr_path}.keyTime")
        self.key_value = output_type(f"{attr_path}.keyValue")
