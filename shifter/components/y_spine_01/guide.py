from functools import partial

from maya.app.general.mayaMixin import MayaQDockWidget, MayaQWidgetDockableMixin  # type: ignore
from mgear.core import pyqt, transform, vector
from mgear.shifter.component import guide
from mgear.vendor.Qt import QtCore, QtWidgets  # type: ignore

from . import settingsUI as sui

# guide info
AUTHOR = "Michael Harmon"
URL = "https://github.com/michaelharmonart/y-rig"
EMAIL = ""
VERSION = [1, 0, 0]
TYPE = "y_spine_01"
NAME = "spine"
DESCRIPTION = """Custom y-rig hybrid FK IK spine."""

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

    joint_names_description = ["pelvis", "spine_##"]

    def postInit(self):
        """Initialize the position for the guide"""
        self.save_transform = [
            "root",
            "spineBase",
            "hipPivot",
            "tan0",
            "tan1",
            "chestPivot",
            "spineTop",
            "chest",
        ]
        self.save_blade = ["blade"]

    def addObjects(self):
        """Add the Guide Root, blade and locators"""

        self.root = self.addRoot()
        vTemp = transform.getOffsetPosition(self.root, [0, 0.5, 0])
        self.spineBase = self.addLoc("spineBase", self.root, vTemp)
        vTemp = transform.getOffsetPosition(self.root, [0, 4, 0])
        self.spineTop = self.addLoc("spineTop", self.spineBase, vTemp)
        vTemp = transform.getOffsetPosition(self.root, [0, 5, 0])
        self.chest = self.addLoc("chest", self.spineTop, vTemp)

        v_hip_pivot = vector.linearlyInterpolate(
            self.spineBase.getTranslation(space="world"),  # type: ignore
            self.spineTop.getTranslation(space="world"),  # type: ignore
            1 / 3,
        )
        self.hipPivot = self.addLoc("hipPivot", self.spineBase, v_hip_pivot)

        vTan0 = vector.linearlyInterpolate(
            self.spineBase.getTranslation(space="world"),  # type: ignore
            self.spineTop.getTranslation(space="world"),  # type: ignore
            1 / 3,
        )
        self.tan0 = self.addLoc("tan0", self.spineBase, vTan0)

        vTan1 = vector.linearlyInterpolate(
            self.spineTop.getTranslation(space="world"),  # type: ignore
            self.spineBase.getTranslation(space="world"),  # type: ignore
            1 / 3,
        )
        self.tan1 = self.addLoc("tan1", self.spineTop, vTan1)

        v_chest_pivot = vector.linearlyInterpolate(
            self.spineBase.getTranslation(space="world"),  # type: ignore
            self.spineTop.getTranslation(space="world"),  # type: ignore
            2 / 3,
        )
        self.chestPivot = self.addLoc("chestPivot", self.spineBase, v_chest_pivot)

        self.blade = self.addBlade("blade", self.root, self.spineTop)

        # spine curve
        self.disp_crv_hip = self.addDispCurve("crvHip", [self.root, self.spineBase])
        self.disp_crv_chst = self.addDispCurve("crvChest", [self.spineTop, self.chest])
        centers = [self.spineBase, self.tan0, self.tan1, self.spineTop]
        self.dispcrv = self.addDispCurve("crv", centers, 3)
        self.dispcrv.attr("lineWidth").set(5)  # type: ignore

        # tangent handles
        self.disp_tancrv0 = self.addDispCurve("crvTan0", [self.spineBase, self.tan0])
        self.disp_tancrv1 = self.addDispCurve("crvTan1", [self.spineTop, self.tan1])

    def addParameters(self):
        """Add the configurations settings"""

        # Default values
        self.pDivision = self.addParam("division", "long", 4, 2)
        self.pleafJoints = self.addParam("leafJoints", "bool", False)

        self.pPreserveLength = self.addParam("preserve_length", "double", 1, 0, 1)
        self.pIKWorldOri = self.addParam("IKWorldOri", "bool", True)

        # FCurves
        self.pSt_profile = self.addFCurveParam("st_profile", [[0, 0], [0.5, -1], [1, 0]])

        self.pSq_profile = self.addFCurveParam("sq_profile", [[0, 0], [0.5, 1], [1, 0]])

        self.pUseIndex = self.addParam("useIndex", "bool", False)

        self.pParentJointIndex = self.addParam("parentJointIndex", "long", -1, None, None)

    def get_divisions(self):
        """Returns correct segments divisions"""

        self.divisions = self.root.division.get()  # type: ignore

        return self.divisions


##########################################################
# Setting Page
##########################################################


class settingsTab(QtWidgets.QDialog, sui.Ui_Form):
    """The Component settings UI"""

    def __init__(self, parent=None):
        super(settingsTab, self).__init__(parent)
        self.setupUi(self)


class componentSettings(MayaQWidgetDockableMixin, guide.componentMainSettings):  # type: ignore
    """Create the component setting window"""

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
        self.resize(350, 360)

    def create_componentControls(self):
        return

    def populate_componentControls(self):
        """Populate the controls values.

        Populate the controls values from the custom attributes of the
        component.

        """
        # populate tab
        self.tabs.insertTab(1, self.settingsTab, "Component Settings")

        # populate component settings
        self.settingsTab.softness_slider.setValue(int(self.root.attr("softness").get() * 100))
        self.settingsTab.position_spinBox.setValue(int(self.root.attr("position").get() * 100))
        self.settingsTab.position_slider.setValue(int(self.root.attr("position").get() * 100))
        self.settingsTab.lockOri_pelvis_spinBox.setValue(
            int(self.root.attr("lock_ori_pelvis").get() * 100)
        )
        self.settingsTab.lockOri_pelvis_slider.setValue(
            int(self.root.attr("lock_ori_pelvis").get() * 100)
        )
        self.settingsTab.lockOri_chest_spinBox.setValue(
            int(self.root.attr("lock_ori_chest").get() * 100)
        )
        self.settingsTab.lockOri_chest_slider.setValue(
            int(self.root.attr("lock_ori_chest").get() * 100)
        )
        self.settingsTab.softness_spinBox.setValue(int(self.root.attr("softness").get() * 100))
        self.settingsTab.maxStretch_spinBox.setValue(self.root.attr("maxstretch").get())
        self.settingsTab.maxSquash_spinBox.setValue(self.root.attr("maxsquash").get())
        self.settingsTab.division_spinBox.setValue(self.root.attr("division").get())
        self.populateCheck(self.settingsTab.autoBend_checkBox, "autoBend")
        self.populateCheck(self.settingsTab.IKWorldOri_checkBox, "IKWorldOri")
        self.populateCheck(self.settingsTab.centralTangent_checkBox, "centralTangent")
        self.populateCheck(self.settingsTab.leafJoints_checkBox, "leafJoints")

    def create_componentLayout(self):
        self.settings_layout = QtWidgets.QVBoxLayout()
        self.settings_layout.addWidget(self.tabs)
        self.settings_layout.addWidget(self.close_button)

        self.setLayout(self.settings_layout)

    def create_componentConnections(self):
        self.settingsTab.softness_slider.valueChanged.connect(
            partial(self.updateSlider, self.settingsTab.softness_slider, "softness")
        )
        self.settingsTab.softness_spinBox.valueChanged.connect(
            partial(
                self.updateSlider,
                self.settingsTab.softness_spinBox,
                "softness",
            )
        )
        self.settingsTab.position_slider.valueChanged.connect(
            partial(self.updateSlider, self.settingsTab.position_slider, "position")
        )
        self.settingsTab.position_spinBox.valueChanged.connect(
            partial(
                self.updateSlider,
                self.settingsTab.position_spinBox,
                "position",
            )
        )
        self.settingsTab.lockOri_pelvis_slider.valueChanged.connect(
            partial(
                self.updateSlider,
                self.settingsTab.lockOri_pelvis_slider,
                "lock_ori_pelvis",
            )
        )
        self.settingsTab.lockOri_pelvis_spinBox.valueChanged.connect(
            partial(
                self.updateSlider,
                self.settingsTab.lockOri_pelvis_spinBox,
                "lock_ori_pelvis",
            )
        )
        self.settingsTab.lockOri_chest_slider.valueChanged.connect(
            partial(
                self.updateSlider,
                self.settingsTab.lockOri_chest_slider,
                "lock_ori_chest",
            )
        )
        self.settingsTab.lockOri_chest_spinBox.valueChanged.connect(
            partial(
                self.updateSlider,
                self.settingsTab.lockOri_chest_spinBox,
                "lock_ori_chest",
            )
        )
        self.settingsTab.maxStretch_spinBox.valueChanged.connect(
            partial(
                self.updateSpinBox,
                self.settingsTab.maxStretch_spinBox,
                "maxstretch",
            )
        )
        self.settingsTab.maxSquash_spinBox.valueChanged.connect(
            partial(
                self.updateSpinBox,
                self.settingsTab.maxSquash_spinBox,
                "maxsquash",
            )
        )
        self.settingsTab.division_spinBox.valueChanged.connect(
            partial(
                self.updateSpinBox,
                self.settingsTab.division_spinBox,
                "division",
            )
        )
        self.settingsTab.autoBend_checkBox.stateChanged.connect(
            partial(
                self.updateCheck,
                self.settingsTab.autoBend_checkBox,
                "autoBend",
            )
        )
        self.settingsTab.IKWorldOri_checkBox.stateChanged.connect(
            partial(
                self.updateCheck,
                self.settingsTab.IKWorldOri_checkBox,
                "IKWorldOri",
            )
        )
        self.settingsTab.centralTangent_checkBox.stateChanged.connect(
            partial(
                self.updateCheck,
                self.settingsTab.centralTangent_checkBox,
                "centralTangent",
            )
        )
        self.settingsTab.squashStretchProfile_pushButton.clicked.connect(self.setProfile)

        self.settingsTab.leafJoints_checkBox.stateChanged.connect(
            partial(
                self.updateCheck,
                self.settingsTab.leafJoints_checkBox,
                "leafJoints",
            )
        )

    def dockCloseEventTriggered(self):
        pyqt.deleteInstances(self, MayaQDockWidget)
