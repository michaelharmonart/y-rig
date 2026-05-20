"""Guide Chain 01 module"""

from functools import partial

from mgear.shifter.component import guide
from mgear.core import pyqt
from mgear.vendor.Qt import QtWidgets, QtCore  # type: ignore

from maya.app.general.mayaMixin import MayaQWidgetDockableMixin  # type: ignore
from maya.app.general.mayaMixin import MayaQDockWidget  # type: ignore
from mgear.core import transform


from . import settingsUI as sui


# guide info
AUTHOR = "Collin Verbanatz"
URL = "https://github.com/michaelharmonart/y-rig"
EMAIL = ", "
VERSION = [1, 0, 1]
TYPE = "y_wheel_01"
NAME = "wheel"
DESCRIPTION = "Simple Wheel component."

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
    connectors = ["y_car_body_01"]

    def postInit(self):
        self.save_transform = [
            "root",
            "ball",
            "steer",
            "wheel",
            "width",
            "lower_arm",
            "lower_ball",
            "upper_arm",
            "front_arm",
            "upperSpring",
            "lowerSpring",
        ]

    def addObjects(self):
        self.root = self.addRoot()
        vTemp = transform.getOffsetPosition(self.root, [53.130, 0, 121.403])
        self.ball = self.addLoc("ball", self.root, vTemp)
        vTemp = transform.getOffsetPosition(self.root, [53.130, 0, 94.763])
        self.steer = self.addLoc("steer", self.root, vTemp)
        vTemp = transform.getOffsetPosition(self.root, [88.130, 0, 121.403])
        self.wheel = self.addLoc("wheel", self.root, vTemp)
        vTemp = transform.getOffsetPosition(self.root, [8, 0, 121.403])
        self.width = self.addLoc("width", self.root, vTemp)
        vTemp = transform.getOffsetPosition(self.width, [10.642, -8.181, 0.044])
        self.lower_arm = self.addLoc("lower_arm", self.width, vTemp)
        vTemp = transform.getOffsetPosition(self.lower_arm, [22.408, -5.495, 0])
        self.lower_ball = self.addLoc("lower_ball", self.lower_arm, vTemp)
        vTemp = transform.getOffsetPosition(self.width, [10.646, 22.408, 0.044])
        self.upper_arm = self.addLoc("upper_arm", self.width, vTemp)
        vTemp = transform.getOffsetPosition(self.width, [10.646, 0, 0.044])
        self.front_arm = self.addLoc("front_arm", self.width, vTemp)
        vTemp = transform.getOffsetPosition(self.upper_arm, [0, 0, 14.53])
        self.upperSpring = self.addLoc("upperSpring", self.upper_arm, vTemp)
        vTemp = transform.getOffsetPosition(self.lower_ball, [0, 0, 14])
        self.lowerSpring = self.addLoc("lowerSpring", self.lower_ball, vTemp)

        centers = [
            self.root,
            self.ball,
            self.steer,
            self.wheel,
            self.width,
            self.lower_arm,
            self.lower_ball,
            self.upper_arm,
            self.front_arm,
            self.upperSpring,
            self.lowerSpring,
        ]
        self.dispcrv = self.addDispCurve("crv", centers)

    def addParameters(self):
        """Add the configurations settings"""

        self.pType = self.addParam("mode", "long", 0, 0)
        self.pBlend = self.addParam("blend", "double", 1, 0, 1)
        self.pNeutralPose = self.addParam("neutralpose", "bool", True)
        self.pIkRefArray = self.addParam("ikrefarray", "string", "")
        self.pUseIndex = self.addParam("useIndex", "bool", False)
        self.pParentJointIndex = self.addParam("parentJointIndex", "long", -1, None, None)
        self.pWheelType = self.addParam("wheelType", "long", 0, 0, 1)

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
        """Populate Controls

        Populate the controls values from the custom attributes of the
        component.

        """
        # populate tab
        self.tabs.insertTab(1, self.settingsTab, "Component Settings")

        # populate component settings
        self.settingsTab.ikfk_slider.setValue(int(self.root.attr("blend").get() * 100))
        self.settingsTab.ikfk_spinBox.setValue(int(self.root.attr("blend").get() * 100))
        self.settingsTab.mode_comboBox.setCurrentIndex(self.root.attr("mode").get())

        for cnx in Guide.connectors:
            self.mainSettingsTab.connector_comboBox.addItem(cnx)

        if self.root.attr("neutralpose").get():
            self.settingsTab.neutralPose_checkBox.setCheckState(QtCore.Qt.Checked)
        else:
            self.settingsTab.neutralPose_checkBox.setCheckState(QtCore.Qt.Unchecked)

        ikRefArrayItems = self.root.attr("ikrefarray").get().split(",")
        for item in ikRefArrayItems:
            self.settingsTab.ikRefArray_listWidget.addItem(item)

        self.settingsTab.wheelType_comboBox.currentIndexChanged.connect(
            partial(self.updateComboBox, self.settingsTab.wheelType_comboBox, "wheelType")
        )

    def create_componentLayout(self):
        self.settings_layout = QtWidgets.QVBoxLayout()
        self.settings_layout.addWidget(self.tabs)
        self.settings_layout.addWidget(self.close_button)

        self.setLayout(self.settings_layout)

    def create_componentConnections(self):
        self.settingsTab.ikfk_slider.valueChanged.connect(
            partial(self.updateSlider, self.settingsTab.ikfk_slider, "blend")
        )
        self.settingsTab.ikfk_spinBox.valueChanged.connect(
            partial(self.updateSlider, self.settingsTab.ikfk_spinBox, "blend")
        )

        self.settingsTab.mode_comboBox.currentIndexChanged.connect(
            partial(self.updateComboBox, self.settingsTab.mode_comboBox, "mode")
        )

        self.settingsTab.neutralPose_checkBox.stateChanged.connect(
            partial(self.updateCheck, self.settingsTab.neutralPose_checkBox, "neutralpose")
        )

        self.settingsTab.ikRefArrayAdd_pushButton.clicked.connect(
            partial(self.addItem2listWidget, self.settingsTab.ikRefArray_listWidget, "ikrefarray")
        )
        self.settingsTab.ikRefArrayRemove_pushButton.clicked.connect(
            partial(
                self.removeSelectedFromListWidget,
                self.settingsTab.ikRefArray_listWidget,
                "ikrefarray",
            )
        )
        self.settingsTab.ikRefArray_listWidget.installEventFilter(self)

        partial(
            self.updateComboBox,
            self.settingsTab.wheelType_comboBox,
            "wheelType",
        )

    def eventFilter(self, sender, event):
        if event.type() == QtCore.QEvent.ChildRemoved:
            if sender == self.settingsTab.ikRefArray_listWidget:
                self.updateListAttr(sender, "ikrefarray")
            return True
        else:
            return QtWidgets.QDialog.eventFilter(self, sender, event)

    def dockCloseEventTriggered(self):
        pyqt.deleteInstances(self, MayaQDockWidget)
