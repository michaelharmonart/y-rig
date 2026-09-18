from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Qt import QtGui, QtWidgets

from yrig.build.nxt_api import setup_rig_build_nxt_layer
from yrig.color.convert import byte_color_to_hex, linear_to_srgb_color, rgb_to_byte_color
from yrig.color.palette import random_color_fixed_lightness_chroma
from yrig.ui import maya_main_window


def setup_rig_build_file(filepath: Path, rig_path: Path) -> bool:
    random_color = byte_color_to_hex(
        rgb_to_byte_color(linear_to_srgb_color(random_color_fixed_lightness_chroma()))
    )
    config = run_rig_build_setup_wizard(
        filepath, name=rig_path.name, rig_path=rig_path, color=random_color
    )
    if config is None:
        return False
    setup_rig_build_nxt_layer(
        filepath,
        rig_path=Path(config.rig_path),
        inherits=config.inherits,
        name=config.name,
        color=config.color,
    )
    return True


def run_rig_build_setup_wizard(
    filepath: Path, name: str, rig_path: Path, color: str
) -> RigBuildLayerConfig | None:
    wizard = RigBuildSetupWizard(
        filepath, name=name, rig_path=rig_path.as_posix(), color=color, parent=maya_main_window()
    )

    if wizard.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None

    return wizard.config()


@dataclass(frozen=True)
class RigBuildLayerConfig:
    filepath: Path
    rig_path: Path
    inherits: Path | None
    name: str
    color: str


class RigBuildSetupWizard(QtWidgets.QWizard):
    def __init__(
        self,
        filepath: Path,
        name: str = "",
        rig_path: str = "",
        color: str = "#1e84d6",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Create Rig Build Layer")

        page = QtWidgets.QWizardPage()
        form = QtWidgets.QFormLayout(page)

        self.filepath = QtWidgets.QLineEdit(filepath.as_posix())
        self.rig_path = QtWidgets.QLineEdit(rig_path)
        self.inherits_path = QtWidgets.QLineEdit()

        self.name = QtWidgets.QLineEdit(name)
        self.color = QtGui.QColor(color)
        self.color_button = QtWidgets.QPushButton()
        self._update_color_button()
        self.color_button.clicked.connect(self._choose_color)  # type: ignore

        form.addRow("Layer:", self.filepath)
        form.addRow("Name:", self.name)
        form.addRow("Rig Path:", self.rig_path)
        form.addRow("Inherits:", self.inherits_path)
        form.addRow("Color:", self.color_button)

        self.setPage(0, page)

    def _choose_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(
            self.color,
            self,
            "Choose Layer Color",
        )
        if color.isValid():
            self.color = color
            self._update_color_button()

    def _update_color_button(self) -> None:
        self.color_button.setText(self.color.name())
        self.color_button.setStyleSheet(f"background-color: {self.color.name()};")

    def config(self) -> RigBuildLayerConfig:
        inherits = self.inherits_path.text().strip()

        return RigBuildLayerConfig(
            filepath=Path(self.filepath.text()),
            rig_path=Path(self.rig_path.text()),
            inherits=Path(inherits) if inherits else None,
            name=self.name.text().strip(),
            color=self.color.name(),
        )
