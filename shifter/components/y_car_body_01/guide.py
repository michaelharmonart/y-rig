"""Guide Chain 01 module"""

from mgear.shifter.component import guide
from mgear.core import pyqt
from mgear.vendor.Qt import QtWidgets, QtCore  # type: ignore

from maya.app.general.mayaMixin import MayaQWidgetDockableMixin  # type: ignore
from maya.app.general.mayaMixin import MayaQDockWidget  # type: ignore
from mgear.core import transform


from . import settingsUI as sui
import importlib

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
        self.save_transform = ["root", "chassis"]
        # self.connectors = ["y_wheel_01"]

    def addObjects(self):
        self.root = self.addRoot()
        vTemp = transform.getOffsetPosition(self.root, [0, 35, 0])
        self.chassis = self.addLoc("chassis", self.root, vTemp)

        centers = [self.root, self.chassis]
        self.dispcrv = self.addDispCurve("crv", centers)

    def addParameters(self):
        """Add the configurations settings"""

        # ===== REQUIRED BY MGEAR (DO NOT REMOVE) =====
        self.pUseIndex = self.addParam("useIndex", "bool", False)
        self.pParentJointIndex = self.addParam("parentJointIndex", "long", -1, None, None)

        # ===== YOUR CUSTOM PARAMS =====
        self.pSide = self.addParam("side", "long", 0, 0, 2)
        self.pWheels = self.addParam("wheels", "string", "")

        # TODO: if have IK or IK/FK lock the axis position to
        # force 2D Planar IK solver
        # Create a a method to lock and unlock while changing
        # options in the PYSIDE component Settings


##########################################################
# Setting Page
##########################################################


class settingsTab(QtWidgets.QDialog, sui.Ui_Form):
    def __init__(self, parent=None):
        super(settingsTab, self).__init__(parent)
        self.setupUi(self)


class componentSettings(MayaQWidgetDockableMixin, guide.componentMainSettings):  # type: ignore
    def __init__(self, parent=None):
        self.toolName = TYPE
        # Delete old instances of the componet settings window.
        pyqt.deleteInstances(self, MayaQDockWidget)

        super(componentSettings, self).__init__(parent=parent)
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

    def updateName(self, text):
        if text:
            self.root.rename(text)

    def dockCloseEventTriggered(self):
        pyqt.deleteInstances(self, MayaQDockWidget)
