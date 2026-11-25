from PySide6 import QtCore, QtWidgets

from .models import AnimationSequence, FrameRange


def create_frame_spinbox(value: int = 0) -> QtWidgets.QSpinBox:
    spinbox = QtWidgets.QSpinBox()
    spinbox.setRange(-1_000_000, 1_000_000)
    spinbox.setValue(value)
    return spinbox


class AnimationSequenceWidget(QtWidgets.QWidget):
    remove_requested = QtCore.Signal(object)
    changed = QtCore.Signal()

    def __init__(
        self,
        sequence: AnimationSequence,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self._build_ui()
        self._connect_signals()
        self.set_sequence(sequence)

    def _build_ui(self) -> None:
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.enabled = QtWidgets.QCheckBox()

        self.suffix = QtWidgets.QLineEdit()
        self.suffix.setPlaceholderText("Sequence suffix")

        self.start_frame = create_frame_spinbox()
        self.end_frame = create_frame_spinbox()

        self.remove_button = QtWidgets.QPushButton("Remove")

        self.preview = QtWidgets.QLabel()

        layout.addWidget(self.enabled, 0, 0)
        layout.addWidget(self.suffix, 0, 1, 1, 3)
        layout.addWidget(self.remove_button, 0, 4)

        layout.addWidget(QtWidgets.QLabel("Start"), 1, 1)
        layout.addWidget(self.start_frame, 1, 2)

        layout.addWidget(QtWidgets.QLabel("End"), 1, 3)
        layout.addWidget(self.end_frame, 1, 4)

        layout.addWidget(self.preview, 2, 1, 1, 4)

    def _connect_signals(self) -> None:
        self.remove_button.clicked.connect(self._request_removal)  # type:ignore

        self.enabled.toggled.connect(self._emit_changed)  # type:ignore
        self.suffix.textChanged.connect(self._emit_changed)  # type:ignore

        self.start_frame.valueChanged.connect(self._emit_changed)  # type:ignore

        self.end_frame.valueChanged.connect(self._emit_changed)  # type:ignore

    def _emit_changed(self, *_: object) -> None:
        self.changed.emit()  # type: ignore

    def _request_removal(self) -> None:
        self.remove_requested.emit(self)  # type:ignore

    def set_sequence(self, sequence: AnimationSequence) -> None:
        self.enabled.setChecked(sequence.enabled)
        self.suffix.setText(sequence.suffix)

        self.start_frame.setValue(sequence.frame_range.start)

        self.end_frame.setValue(sequence.frame_range.end)

    def get_sequence(self) -> AnimationSequence:
        return AnimationSequence(
            suffix=self.suffix.text().strip(),
            frame_range=FrameRange(
                start=self.start_frame.value(),
                end=self.end_frame.value(),
            ),
            enabled=self.enabled.isChecked(),
        )

    def set_preview(self, filename: str) -> None:
        self.preview.setText(filename)
