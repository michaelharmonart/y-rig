"""Guide Chain 01 module"""

import contextlib
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maya.app.general.mayaMixin import MayaQDockWidget, MayaQWidgetDockableMixin  # type: ignore
else:
    try:
        from maya.app.general.mayaMixin import MayaQDockWidget, MayaQWidgetDockableMixin
    except ImportError:

        class MayaQDockWidget:
            pass

        class MayaQWidgetDockableMixin:
            pass


from mgear.core import pyqt, transform
from mgear.shifter.component import guide
from mgear.vendor.Qt import QtCore, QtWidgets  # type: ignore

from . import settingsUI as sui

importlib.reload(sui)

# guide info
AUTHOR = "Collin Verbanatz"
URL = "https://github.com/michaelharmonart/y-rig"
EMAIL = ", "
VERSION = [1, 0, 1]
TYPE = "y_car_body_01"
NAME = "car_body"
DESCRIPTION = "Simple Car Body component."

##########################################################
# CLASS
##########################################################


class Guide(guide.ComponentGuide):
    """Component Guide Class"""

    compType = TYPE
    compName = NAME
    description = DESCRIPTION

    author = AUTHOR
    url = URL
    email = EMAIL
    version = VERSION

    connectors = ["y_wheel_01"]

    def postInit(self):
        self.save_transform = ["root", "chassis", "left", "right", "front", "back"]
        # self.connectors = ["y_wheel_01"]

    def addObjects(self):
        self.root = self.addRoot()
        vTemp = transform.getOffsetPosition(self.root, [0, 35, 0])
        self.chassis = self.addLoc("chassis", self.root, vTemp)
        leftTemp = transform.getOffsetPosition(self.root, [88.13, 0, 0])
        self.left = self.addLoc("left", self.root, leftTemp)
        rightTemp = transform.getOffsetPosition(self.root, [-88.13, 0, 0])
        self.right = self.addLoc("right", self.root, rightTemp)
        frontTemp = transform.getOffsetPosition(self.root, [0, 0, 121.403])
        self.front = self.addLoc("front", self.root, frontTemp)
        backTemp = transform.getOffsetPosition(self.root, [0, 0, -121.403])
        self.back = self.addLoc("back", self.root, backTemp)
        centers = [self.root, self.chassis, self.left, self.right, self.front, self.back]
        self.dispcrv = self.addDispCurve("crv", centers)

    def addParameters(self):
        """Add the configurations settings"""

        # ===== REQUIRED BY MGEAR (DO NOT REMOVE) =====
        self.pUseIndex = self.addParam("useIndex", "bool", False)
        self.pParentJointIndex = self.addParam("parentJointIndex", "long", -1, None, None)

        # ===== YOUR CUSTOM PARAMS =====
        self.pSide = self.addParam("side", "long", 0, 0, 2)
        self.pWheels = self.addParam("wheels", "string", "")
        self.pWheelRadius = self.addParam("wheelRadius", "double", 35)
        self.pWheelRadius2 = self.addParam("wheelRadius2", "double", 35)

        # TODO: if have IK or IK/FK lock the axis position to
        # force 2D Planar IK solver
        # Create a a method to lock and unlock while changing
        # options in the PYSIDE component Settings


##########################################################
# Setting Page
##########################################################


class settingsTab(QtWidgets.QDialog, sui.Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)


class componentSettings(MayaQWidgetDockableMixin, guide.componentMainSettings):  # type: ignore
    def __init__(self, parent=None):
        self.toolName = TYPE
        # Delete old instances of the componet settings window.
        pyqt.deleteInstances(self, MayaQDockWidget)

        super().__init__(parent=parent)
        self.settingsTab = settingsTab()

        self.setup_componentSettingWindow()
        self.create_componentControls()
        self.populate_componentControls()
        self.create_componentLayout()
        self.create_componentConnections()

    def setup_componentSettingWindow(self):
        self.mayaMainWindow = pyqt.maya_main_window()

        self.setObjectName(self.toolName)
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle(TYPE)
        self.resize(350, 350)

    def create_componentControls(self):
        return

    def populate_componentControls(self):
        # Add tab
        self.tabs.insertTab(1, self.settingsTab, "Component Settings")

        # Populate values
        self.settingsTab.name_lineEdit.setText(self.root.name())
        self.settingsTab.side_comboBox.setCurrentIndex(self.root.attr("side").get())
        self.settingsTab.wheels_lineEdit.setText(self.root.attr("wheels").get())
        self.settingsTab.wheelRadius_lineEdit.setText(str(self.root.attr("wheelRadius").get()))
        self.settingsTab.wheelRadius2_lineEdit.setText(str(self.root.attr("wheelRadius2").get()))

        for cnx in Guide.connectors:
            self.mainSettingsTab.connector_comboBox.addItem(cnx)

    def create_componentLayout(self):
        self.settings_layout = QtWidgets.QVBoxLayout()
        self.settings_layout.addWidget(self.tabs)
        self.settings_layout.addWidget(self.close_button)

        self.setLayout(self.settings_layout)

    def create_componentConnections(self):
        # Name
        self.settingsTab.name_lineEdit.textChanged.connect(self.updateName)

        # Side
        self.settingsTab.side_comboBox.currentIndexChanged.connect(
            lambda val: self.root.attr("side").set(val)
        )

        # Wheels
        self.settingsTab.wheels_lineEdit.textChanged.connect(
            lambda val: self.root.attr("wheels").set(val)
        )

        self.settingsTab.wheelRadius_lineEdit.textChanged.connect(self.updateWheelRadius)
        self.settingsTab.wheelRadius2_lineEdit.textChanged.connect(self.updateWheelRadius2)

    def updateName(self, text):
        if text:
            self.root.rename(text)

    def updateWheelRadius(self, text):
        if text:
            with contextlib.suppress(ValueError):
                self.root.attr("wheelRadius").set(float(text))

    def updateWheelRadius2(self, text):
        if text:
            with contextlib.suppress(ValueError):
                self.root.attr("wheelRadius2").set(float(text))

    def dockCloseEventTriggered(self):
        pyqt.deleteInstances(self, MayaQDockWidget)
