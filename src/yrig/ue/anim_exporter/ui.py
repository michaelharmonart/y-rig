from maya import OpenMayaUI
from PySide6 import QtCore, QtWidgets
from shiboken6 import wrapInstance

from . import context
from .models import (
    AnimationSequence,
    AnimationSettings,
    FrameRange,
)
from .naming import (
    resolve_animation_name,
    resolve_base_name,
    validate_name,
)
from .publisher import (
    create_export_jobs,
    publish_animations,
)
from .storage import load_settings
from .widgets import (
    AnimationSequenceWidget,
    create_frame_spinbox,
)

WINDOW_NAME = "yrig_animation_publisher"


def get_maya_window() -> None:
    pointer = OpenMayaUI.MQtUtil.mainWindow()

    if pointer is not None:
        return wrapInstance(
            int(pointer),
            QtWidgets.QWidget,
        )

    return None


class AnimationPublisherUI(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent or get_maya_window())

        self.setObjectName(WINDOW_NAME)
        self.setWindowTitle("Animation Publisher")
        self.resize(480, 650)

        self.sequence_widgets = []

        self._build_ui()
        self._connect_signals()
        self._load_initial_settings()

    # --------------------------------------------------
    # UI Construction
    # --------------------------------------------------

    def _build_ui(self) -> None:
        self.main_layout = QtWidgets.QVBoxLayout(self)

        self._build_context_section()
        self._build_frame_range_section()
        self._build_sequence_section()
        self._build_naming_section()
        self._build_export_section()

    def _build_context_section(self) -> None:
        group = QtWidgets.QGroupBox("Shot Information")
        layout = QtWidgets.QFormLayout(group)

        # Shot
        self.shot_name = QtWidgets.QLineEdit()
        self.shot_name.setReadOnly(True)

        self.reload_shot = QtWidgets.QPushButton("Reload")

        shot_layout = QtWidgets.QHBoxLayout()
        shot_layout.addWidget(self.shot_name, 1)
        shot_layout.addWidget(self.reload_shot)

        # Rig
        self.rig_selector = QtWidgets.QComboBox()
        self.reload_rigs = QtWidgets.QPushButton("Reload")

        rig_layout = QtWidgets.QHBoxLayout()
        rig_layout.addWidget(self.rig_selector, 1)
        rig_layout.addWidget(self.reload_rigs)

        layout.addRow("Shot", shot_layout)
        layout.addRow("Rig", rig_layout)

        self.main_layout.addWidget(group)

    def _build_frame_range_section(self) -> None:
        group = QtWidgets.QGroupBox("Default Frame Range")
        layout = QtWidgets.QGridLayout(group)

        self.start_frame = create_frame_spinbox()
        self.end_frame = create_frame_spinbox()

        self.timeline_button = QtWidgets.QPushButton("From Timeline")

        self.pipeline_button = QtWidgets.QPushButton("From Pipeline")

        layout.addWidget(QtWidgets.QLabel("Start"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("End"), 0, 1)

        layout.addWidget(self.start_frame, 1, 0)
        layout.addWidget(self.end_frame, 1, 1)

        layout.addWidget(self.timeline_button, 2, 0)
        layout.addWidget(self.pipeline_button, 2, 1)

        self.main_layout.addWidget(group)

    def _build_sequence_section(self) -> None:
        group = QtWidgets.QGroupBox("Animation Sequences")
        layout = QtWidgets.QVBoxLayout(group)

        self.export_full = QtWidgets.QCheckBox("Export full animation")

        self.full_preview = QtWidgets.QLabel()

        self.add_sequence_button = QtWidgets.QPushButton("Add Sequence")

        layout.addWidget(self.export_full)
        layout.addWidget(self.full_preview)
        layout.addWidget(self.add_sequence_button)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        container = QtWidgets.QWidget()

        self.sequence_layout = QtWidgets.QVBoxLayout(container)
        self.sequence_layout.setAlignment(QtCore.Qt.AlignTop)  # type:ignore

        scroll.setWidget(container)

        layout.addWidget(scroll)

        self.main_layout.addWidget(group, 1)

    def _build_naming_section(self) -> None:
        group = QtWidgets.QGroupBox("Animation Naming")
        layout = QtWidgets.QVBoxLayout(group)

        self.override_name = QtWidgets.QCheckBox("Override shot name")

        self.custom_name = QtWidgets.QLineEdit()
        self.custom_name.setEnabled(False)

        layout.addWidget(self.override_name)
        layout.addWidget(self.custom_name)

        self.main_layout.addWidget(group)

    def _build_export_section(self) -> None:
        group = QtWidgets.QGroupBox("Export Settings")
        layout = QtWidgets.QVBoxLayout(group)

        self.export_path = QtWidgets.QLineEdit()
        self.export_path.setReadOnly(True)

        self.export_model = QtWidgets.QCheckBox("Include model")

        self.export_button = QtWidgets.QPushButton("Export Animations")

        layout.addWidget(QtWidgets.QLabel("Export Directory"))
        layout.addWidget(self.export_path)
        layout.addWidget(self.export_model)
        layout.addWidget(self.export_button)

        self.main_layout.addWidget(group)

    # --------------------------------------------------
    # Signals
    # --------------------------------------------------

    def _connect_signals(self) -> None:
        self.reload_rigs.clicked.connect(self._refresh_rigs)  # type:ignore

        self.timeline_button.clicked.connect(self._load_timeline_range)  # type:ignore

        self.pipeline_button.clicked.connect(self._load_pipeline_range)  # type:ignore

        self.add_sequence_button.clicked.connect(self._add_sequence)  # type:ignore

        self.override_name.toggled.connect(self._toggle_name_override)  # type:ignore

        self.custom_name.textChanged.connect(self._update_previews)  # type:ignore

        self.export_full.toggled.connect(self._update_previews)  # type:ignore

        self.export_button.clicked.connect(self._publish)  # type:ignore

        self.reload_shot.clicked.connect(self._refresh_shot)  # type:ignore

    # --------------------------------------------------
    # Context
    # --------------------------------------------------

    def _refresh_shot(self) -> None:
        """Refresh shot information from the current context."""

        shot_name = context.get_shot_name()

        self.shot_name.setText(shot_name)

        self.export_path.setText(str(context.get_export_directory()))

        if not self.override_name.isChecked():
            self.custom_name.setText(shot_name)

        self._update_previews()

    def _refresh_rigs(self) -> None:
        current = self.rig_selector.currentText()

        namespaces = context.get_available_rig_namespaces()

        self.rig_selector.clear()
        self.rig_selector.addItems(namespaces)

        index = self.rig_selector.findText(current)

        if index >= 0:
            self.rig_selector.setCurrentIndex(index)

    def _set_frame_range(self, frame_range: FrameRange) -> None:
        self.start_frame.setValue(frame_range.start)
        self.end_frame.setValue(frame_range.end)

    def _get_frame_range(self) -> FrameRange:
        return FrameRange(
            start=self.start_frame.value(),
            end=self.end_frame.value(),
        )

    def _load_timeline_range(self) -> None:
        self._set_frame_range(context.get_timeline_frame_range())

    def _load_pipeline_range(self) -> None:
        try:
            frame_range = context.get_pipeline_frame_range()
            self._set_frame_range(frame_range)

        except Exception as error:
            QtWidgets.QMessageBox.warning(
                self,
                "Pipeline Frame Range",
                str(error),
            )

    # --------------------------------------------------
    # Sequences
    # --------------------------------------------------

    def _add_sequence(
        self,
        checked: bool = False,
        sequence: AnimationSequence | None = None,
    ) -> None:
        if sequence is None:
            existing = {widget.get_sequence().suffix for widget in self.sequence_widgets}

            index = 1

            while f"sequence_{index:02d}" in existing:
                index += 1

            sequence = AnimationSequence(
                suffix=f"sequence_{index:02d}",
                frame_range=self._get_frame_range(),
            )

        widget = AnimationSequenceWidget(sequence)

        widget.remove_requested.connect(self._remove_sequence)  # type:ignore

        widget.changed.connect(self._update_previews)  # type:ignore

        self.sequence_widgets.append(widget)
        self.sequence_layout.addWidget(widget)

        self._update_previews()

    def _remove_sequence(self, widget: AnimationSequenceWidget) -> None:
        self.sequence_widgets.remove(widget)

        self.sequence_layout.removeWidget(widget)
        widget.deleteLater()

        self._update_previews()

    def _get_sequences(self) -> list[AnimationSequence]:
        return [widget.get_sequence() for widget in self.sequence_widgets]

    # --------------------------------------------------
    # Naming
    # --------------------------------------------------

    def _toggle_name_override(self, enabled: bool) -> None:
        self.custom_name.setEnabled(enabled)

        if enabled and not self.custom_name.text():
            self.custom_name.setText(self.shot_name.text())

        self._update_previews()

    def _update_previews(self, *_) -> None:
        settings = self._collect_settings()

        try:
            base_name = resolve_base_name(
                self.shot_name.text(),
                settings,
            )

        except ValueError:
            base_name = None

        self.full_preview.setText(f"{base_name}.fbx" if base_name else "Invalid animation name")

        self.full_preview.setEnabled(settings.export_full)

        for widget in self.sequence_widgets:
            sequence = widget.get_sequence()

            try:
                suffix = validate_name(sequence.suffix)

                if base_name is None:
                    raise ValueError("Invalid base name")

                name = resolve_animation_name(
                    base_name,
                    suffix,
                )

                widget.set_preview(f"{name}.fbx")

            except ValueError:
                widget.set_preview("Invalid animation name")

        count = int(settings.export_full)

        count += sum(sequence.enabled for sequence in settings.sequences)

        self.export_button.setText(f"Export {count} Animation" + ("s" if count != 1 else ""))

        self.export_button.setEnabled(count > 0)

    # --------------------------------------------------
    # Settings
    # --------------------------------------------------

    def _collect_settings(self) -> AnimationSettings:
        return AnimationSettings(
            frame_range=self._get_frame_range(),
            sequences=self._get_sequences(),
            export_full=self.export_full.isChecked(),
            export_model=self.export_model.isChecked(),
            override_name=self.override_name.isChecked(),
            custom_name=self.custom_name.text(),
            namespace=self.rig_selector.currentText(),
        )

    def _load_initial_settings(self) -> None:
        self._refresh_shot()

        self._refresh_rigs()

        settings = load_settings()

        if settings is None:
            settings = AnimationSettings(frame_range=context.get_default_frame_range())

        self._set_frame_range(settings.frame_range)

        self.export_full.setChecked(settings.export_full)
        self.export_model.setChecked(settings.export_model)

        self.custom_name.setText(settings.custom_name or self.shot_name.text())

        self.override_name.setChecked(settings.override_name)

        index = self.rig_selector.findText(settings.namespace)

        if index >= 0:
            self.rig_selector.setCurrentIndex(index)
        elif settings.namespace:
            self.rig_selector.setCurrentIndex(-1)

        for sequence in settings.sequences:
            self._add_sequence(sequence=sequence)

        self._update_previews()

    # --------------------------------------------------
    # Publishing
    # --------------------------------------------------

    def _publish(self) -> None:
        settings = self._collect_settings()
        shot_name = self.shot_name.text()
        directory = context.get_export_directory()

        try:
            jobs = create_export_jobs(
                settings=settings,
                shot_name=shot_name,
                export_directory=directory,
            )

            existing = [job.path.name for job in jobs if job.path.exists()]

            overwrite = False

            if existing:
                response = QtWidgets.QMessageBox.question(
                    self,
                    "Overwrite Animations",
                    "These animations already exist:\n\n"
                    + "\n".join(existing)
                    + "\n\nOverwrite existing files?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,  # type:ignore
                    QtWidgets.QMessageBox.No,  # type:ignore
                )

                if response != QtWidgets.QMessageBox.Yes:  # type:ignore
                    return

                overwrite = True

            paths = publish_animations(
                settings=settings,
                shot_name=shot_name,
                export_directory=directory,
                overwrite=overwrite,
            )

        except Exception as error:
            QtWidgets.QMessageBox.critical(
                self,
                "Animation Export Failed",
                str(error),
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Animation Export Complete",
            f"Successfully exported {len(paths)} animations:\n\n"
            + "\n".join(str(path) for path in paths)
            + "\n\nSave the Maya scene to retain export settings.",
        )


_window = None


def show() -> AnimationPublisherUI:
    global _window

    if _window is not None:
        _window.close()
        _window.deleteLater()

    _window = AnimationPublisherUI()
    _window.show()

    return _window
