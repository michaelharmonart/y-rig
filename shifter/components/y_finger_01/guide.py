"""Guide Y Finger 01 module"""

from functools import partial

from maya.app.general.mayaMixin import MayaQDockWidget, MayaQWidgetDockableMixin  # type: ignore
from mgear.core import pyqt
from mgear.shifter.component import guide
from mgear.vendor.Qt import QtCore, QtWidgets  # type: ignore

from . import settingsUI as sui

# guide info
AUTHOR = "y-rig"
URL = "https://github.com/michaelharmonart/y-rig"
EMAIL = ""
VERSION = [1, 0, 0]
TYPE = "y_finger_01"
NAME = "finger"
DESCRIPTION = "Chain component for finger controls with global roll control"

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

    def postInit(self):
        """Initialize the position for the guide"""
        self.save_transform = ["root", "#_loc"]
        self.save_blade = ["blade"]
        self.addMinMax("#_loc", 1, -1)

    def addObjects(self):
        """Add the Guide Root, blade and locators"""

        self.root = self.addRoot()
        self.locs = self.addLocMulti("#_loc", self.root)
        self.blade = self.addBlade("blade", self.root, self.locs[0])

        centers = [self.root]
        centers.extend(self.locs)
        self.dispcrv = self.addDispCurve("crv", centers)

    def addParameters(self):
        """Add the configurations settings"""

        self.pType = self.addParam("mode", "long", 0, 0)
        self.pBlend = self.addParam("blend", "double", 1, 0, 1)
        self.pNeutralPose = self.addParam("neutralpose", "bool", True)
        self.pIkRefArray = self.addParam("ikrefarray", "string", "")
        self.pUseIndex = self.addParam("useIndex", "bool", False)
        self.pParentJointIndex = self.addParam("parentJointIndex", "long", -1, None, None)
        self.pNeutralPose = self.addParam("neutralpose", "bool", True)
        self.pIsThumb = self.addParam("is_thumb", "bool", False)
        self.pIkRefArray = self.addParam("ikrefarray", "string", "")


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
        self.tabs.insertTab(1, self.settingsTab, "Component Settings")

        self.settingsTab.ikfk_slider.setValue(int(self.root.attr("blend").get() * 100))
        self.settingsTab.ikfk_spinBox.setValue(int(self.root.attr("blend").get() * 100))
        self.settingsTab.mode_comboBox.setCurrentIndex(self.root.attr("mode").get())

        if self.root.attr("neutralpose").get():
            self.settingsTab.neutralPose_checkBox.setCheckState(QtCore.Qt.Checked)
        else:
            self.settingsTab.neutralPose_checkBox.setCheckState(QtCore.Qt.Unchecked)

        ikRefArrayItems = self.root.attr("ikrefarray").get().split(",")
        for item in ikRefArrayItems:
            self.settingsTab.ikRefArray_listWidget.addItem(item)

        if self.root.attr("is_thumb").get():
            self.settingsTab.isThumb_checkBox.setCheckState(QtCore.Qt.Checked)
        else:
            self.settingsTab.isThumb_checkBox.setCheckState(QtCore.Qt.Unchecked)

    def create_componentLayout(self):
        self.settings_layout = QtWidgets.QVBoxLayout()
        self.settings_layout.addWidget(self.tabs)
        self.settings_layout.addWidget(self.close_button)
        self.settingsTab.isThumb_checkBox.stateChanged.connect(
            partial(
                self.updateCheck,
                self.settingsTab.isThumb_checkBox,
                "is_thumb",
            )
        )

        self.setLayout(self.settings_layout)

    def create_componentConnections(self):
        self.settingsTab.ikfk_slider.valueChanged.connect(
            partial(self.updateSlider, self.settingsTab.ikfk_slider, "blend")
        )
        self.settingsTab.ikfk_spinBox.valueChanged.connect(
            partial(self.updateSlider, self.settingsTab.ikfk_spinBox, "blend")
        )
        self.settingsTab.mode_comboBox.currentIndexChanged.connect(
            partial(
                self.updateComboBox,
                self.settingsTab.mode_comboBox,
                "mode",
            )
        )
        self.settingsTab.neutralPose_checkBox.stateChanged.connect(
            partial(
                self.updateCheck,
                self.settingsTab.neutralPose_checkBox,
                "neutralpose",
            )
        )
        self.settingsTab.ikRefArrayAdd_pushButton.clicked.connect(
            partial(
                self.addItem2listWidget,
                self.settingsTab.ikRefArray_listWidget,
                "ikrefarray",
            )
        )
        self.settingsTab.ikRefArrayRemove_pushButton.clicked.connect(
            partial(
                self.removeSelectedFromListWidget,
                self.settingsTab.ikRefArray_listWidget,
                "ikrefarray",
            )
        )
        self.settingsTab.ikRefArray_listWidget.installEventFilter(self)

    def eventFilter(self, sender, event):
        if event.type() == QtCore.QEvent.ChildRemoved:
            if sender == self.settingsTab.ikRefArray_listWidget:
                self.updateListAttr(sender, "ikrefarray")
            return True
        else:
            return QtWidgets.QDialog.eventFilter(self, sender, event)

    def dockCloseEventTriggered(self):
        pyqt.deleteInstances(self, MayaQDockWidget)
