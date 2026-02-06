# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'settingsUImkMJBL.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from Qt.QtCore import QCoreApplication, QMetaObject, QSize, Qt
from Qt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QVBoxLayout,
)


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName("Form")
        Form.resize(364, 1028)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.formLayout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.ikfk_label = QLabel(self.groupBox)
        self.ikfk_label.setObjectName("ikfk_label")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.ikfk_label)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.ikfk_slider = QSlider(self.groupBox)
        self.ikfk_slider.setObjectName("ikfk_slider")
        self.ikfk_slider.setMinimumSize(QSize(0, 15))
        self.ikfk_slider.setMaximum(100)
        self.ikfk_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_3.addWidget(self.ikfk_slider)

        self.ikfk_spinBox = QSpinBox(self.groupBox)
        self.ikfk_spinBox.setObjectName("ikfk_spinBox")
        self.ikfk_spinBox.setMaximum(100)

        self.horizontalLayout_3.addWidget(self.ikfk_spinBox)

        self.formLayout.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_3)

        self.maxStretch_label = QLabel(self.groupBox)
        self.maxStretch_label.setObjectName("maxStretch_label")
        sizePolicy = QSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.maxStretch_label.sizePolicy().hasHeightForWidth())
        self.maxStretch_label.setSizePolicy(sizePolicy)

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.maxStretch_label)

        self.maxStretch_spinBox = QDoubleSpinBox(self.groupBox)
        self.maxStretch_spinBox.setObjectName("maxStretch_spinBox")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.maxStretch_spinBox.sizePolicy().hasHeightForWidth())
        self.maxStretch_spinBox.setSizePolicy(sizePolicy1)
        self.maxStretch_spinBox.setMinimum(1.000000000000000)
        self.maxStretch_spinBox.setMaximum(1000.000000000000000)
        self.maxStretch_spinBox.setValue(100.000000000000000)

        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.maxStretch_spinBox)

        self.verticalLayout.addLayout(self.formLayout)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.divisions_label = QLabel(self.groupBox)
        self.divisions_label.setObjectName("divisions_label")

        self.horizontalLayout.addWidget(self.divisions_label)

        self.div0_spinBox = QSpinBox(self.groupBox)
        self.div0_spinBox.setObjectName("div0_spinBox")
        self.div0_spinBox.setMinimum(0)
        self.div0_spinBox.setValue(2)

        self.horizontalLayout.addWidget(self.div0_spinBox)

        self.div1_spinBox = QSpinBox(self.groupBox)
        self.div1_spinBox.setObjectName("div1_spinBox")
        self.div1_spinBox.setMinimum(0)
        self.div1_spinBox.setValue(2)

        self.horizontalLayout.addWidget(self.div1_spinBox)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.leafJoints_checkBox = QCheckBox(self.groupBox)
        self.leafJoints_checkBox.setObjectName("leafJoints_checkBox")

        self.verticalLayout.addWidget(self.leafJoints_checkBox)

        self.ikTR_checkBox = QCheckBox(self.groupBox)
        self.ikTR_checkBox.setObjectName("ikTR_checkBox")

        self.verticalLayout.addWidget(self.ikTR_checkBox)

        self.mirrorIK_checkBox = QCheckBox(self.groupBox)
        self.mirrorIK_checkBox.setObjectName("mirrorIK_checkBox")

        self.verticalLayout.addWidget(self.mirrorIK_checkBox)

        self.mirrorMid_checkBox = QCheckBox(self.groupBox)
        self.mirrorMid_checkBox.setObjectName("mirrorMid_checkBox")

        self.verticalLayout.addWidget(self.mirrorMid_checkBox)

        self.useBlade_checkBox = QCheckBox(self.groupBox)
        self.useBlade_checkBox.setObjectName("useBlade_checkBox")

        self.verticalLayout.addWidget(self.useBlade_checkBox)

        self.TPoseRest_checkBox = QCheckBox(self.groupBox)
        self.TPoseRest_checkBox.setObjectName("TPoseRest_checkBox")

        self.verticalLayout.addWidget(self.TPoseRest_checkBox)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.squashStretchProfile_pushButton = QPushButton(self.groupBox)
        self.squashStretchProfile_pushButton.setObjectName("squashStretchProfile_pushButton")

        self.horizontalLayout_2.addWidget(self.squashStretchProfile_pushButton)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        self.ikRefArray_groupBox = QGroupBox(Form)
        self.ikRefArray_groupBox.setObjectName("ikRefArray_groupBox")
        self.gridLayout_3 = QGridLayout(self.ikRefArray_groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.ikRefArray_horizontalLayout = QHBoxLayout()
        self.ikRefArray_horizontalLayout.setObjectName("ikRefArray_horizontalLayout")
        self.ikRefArray_verticalLayout_1 = QVBoxLayout()
        self.ikRefArray_verticalLayout_1.setObjectName("ikRefArray_verticalLayout_1")
        self.ikRefArray_listWidget = QListWidget(self.ikRefArray_groupBox)
        self.ikRefArray_listWidget.setObjectName("ikRefArray_listWidget")
        self.ikRefArray_listWidget.setDragDropOverwriteMode(True)
        self.ikRefArray_listWidget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.ikRefArray_listWidget.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.ikRefArray_listWidget.setAlternatingRowColors(True)
        self.ikRefArray_listWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.ikRefArray_listWidget.setSelectionRectVisible(False)

        self.ikRefArray_verticalLayout_1.addWidget(self.ikRefArray_listWidget)

        self.ikRefArray_copyRef_pushButton = QPushButton(self.ikRefArray_groupBox)
        self.ikRefArray_copyRef_pushButton.setObjectName("ikRefArray_copyRef_pushButton")

        self.ikRefArray_verticalLayout_1.addWidget(self.ikRefArray_copyRef_pushButton)

        self.ikRefArray_horizontalLayout.addLayout(self.ikRefArray_verticalLayout_1)

        self.ikRefArray_verticalLayout_2 = QVBoxLayout()
        self.ikRefArray_verticalLayout_2.setObjectName("ikRefArray_verticalLayout_2")
        self.ikRefArrayAdd_pushButton = QPushButton(self.ikRefArray_groupBox)
        self.ikRefArrayAdd_pushButton.setObjectName("ikRefArrayAdd_pushButton")

        self.ikRefArray_verticalLayout_2.addWidget(self.ikRefArrayAdd_pushButton)

        self.ikRefArrayRemove_pushButton = QPushButton(self.ikRefArray_groupBox)
        self.ikRefArrayRemove_pushButton.setObjectName("ikRefArrayRemove_pushButton")

        self.ikRefArray_verticalLayout_2.addWidget(self.ikRefArrayRemove_pushButton)

        self.ikRefArray_verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.ikRefArray_verticalLayout_2.addItem(self.ikRefArray_verticalSpacer)

        self.ikRefArray_horizontalLayout.addLayout(self.ikRefArray_verticalLayout_2)

        self.gridLayout_3.addLayout(self.ikRefArray_horizontalLayout, 0, 0, 1, 1)

        self.gridLayout.addWidget(self.ikRefArray_groupBox, 1, 0, 1, 1)

        self.upvRefArray_groupBox = QGroupBox(Form)
        self.upvRefArray_groupBox.setObjectName("upvRefArray_groupBox")
        self.gridLayout_5 = QGridLayout(self.upvRefArray_groupBox)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.upvRefArray_horizontalLayout = QHBoxLayout()
        self.upvRefArray_horizontalLayout.setObjectName("upvRefArray_horizontalLayout")
        self.upvRefArray_verticalLayout_1 = QVBoxLayout()
        self.upvRefArray_verticalLayout_1.setObjectName("upvRefArray_verticalLayout_1")
        self.upvRefArray_listWidget = QListWidget(self.upvRefArray_groupBox)
        self.upvRefArray_listWidget.setObjectName("upvRefArray_listWidget")
        self.upvRefArray_listWidget.setDragDropOverwriteMode(True)
        self.upvRefArray_listWidget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.upvRefArray_listWidget.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.upvRefArray_listWidget.setAlternatingRowColors(True)
        self.upvRefArray_listWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.upvRefArray_listWidget.setSelectionRectVisible(False)

        self.upvRefArray_verticalLayout_1.addWidget(self.upvRefArray_listWidget)

        self.upvRefArray_copyRef_pushButton = QPushButton(self.upvRefArray_groupBox)
        self.upvRefArray_copyRef_pushButton.setObjectName("upvRefArray_copyRef_pushButton")

        self.upvRefArray_verticalLayout_1.addWidget(self.upvRefArray_copyRef_pushButton)

        self.upvRefArray_horizontalLayout.addLayout(self.upvRefArray_verticalLayout_1)

        self.upvRefArray_verticalLayout_2 = QVBoxLayout()
        self.upvRefArray_verticalLayout_2.setObjectName("upvRefArray_verticalLayout_2")
        self.upvRefArrayAdd_pushButton = QPushButton(self.upvRefArray_groupBox)
        self.upvRefArrayAdd_pushButton.setObjectName("upvRefArrayAdd_pushButton")

        self.upvRefArray_verticalLayout_2.addWidget(self.upvRefArrayAdd_pushButton)

        self.upvRefArrayRemove_pushButton = QPushButton(self.upvRefArray_groupBox)
        self.upvRefArrayRemove_pushButton.setObjectName("upvRefArrayRemove_pushButton")

        self.upvRefArray_verticalLayout_2.addWidget(self.upvRefArrayRemove_pushButton)

        self.upvRefArray_verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.upvRefArray_verticalLayout_2.addItem(self.upvRefArray_verticalSpacer)

        self.upvRefArray_horizontalLayout.addLayout(self.upvRefArray_verticalLayout_2)

        self.gridLayout_5.addLayout(self.upvRefArray_horizontalLayout, 0, 0, 1, 1)

        self.gridLayout.addWidget(self.upvRefArray_groupBox, 2, 0, 1, 1)

        self.pinRefArray_groupBox = QGroupBox(Form)
        self.pinRefArray_groupBox.setObjectName("pinRefArray_groupBox")
        self.gridLayout_4 = QGridLayout(self.pinRefArray_groupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.pinRefArray_horizontalLayout = QHBoxLayout()
        self.pinRefArray_horizontalLayout.setObjectName("pinRefArray_horizontalLayout")
        self.pinRefArray_verticalLayout = QVBoxLayout()
        self.pinRefArray_verticalLayout.setObjectName("pinRefArray_verticalLayout")
        self.pinRefArray_listWidget = QListWidget(self.pinRefArray_groupBox)
        self.pinRefArray_listWidget.setObjectName("pinRefArray_listWidget")
        self.pinRefArray_listWidget.setDragDropOverwriteMode(True)
        self.pinRefArray_listWidget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.pinRefArray_listWidget.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.pinRefArray_listWidget.setAlternatingRowColors(True)
        self.pinRefArray_listWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.pinRefArray_listWidget.setSelectionRectVisible(False)

        self.pinRefArray_verticalLayout.addWidget(self.pinRefArray_listWidget)

        self.pinRefArray_copyRef_pushButton = QPushButton(self.pinRefArray_groupBox)
        self.pinRefArray_copyRef_pushButton.setObjectName("pinRefArray_copyRef_pushButton")

        self.pinRefArray_verticalLayout.addWidget(self.pinRefArray_copyRef_pushButton)

        self.pinRefArray_horizontalLayout.addLayout(self.pinRefArray_verticalLayout)

        self.pinRefArray_verticalLayout_2 = QVBoxLayout()
        self.pinRefArray_verticalLayout_2.setObjectName("pinRefArray_verticalLayout_2")
        self.pinRefArrayAdd_pushButton = QPushButton(self.pinRefArray_groupBox)
        self.pinRefArrayAdd_pushButton.setObjectName("pinRefArrayAdd_pushButton")

        self.pinRefArray_verticalLayout_2.addWidget(self.pinRefArrayAdd_pushButton)

        self.pinRefArrayRemove_pushButton = QPushButton(self.pinRefArray_groupBox)
        self.pinRefArrayRemove_pushButton.setObjectName("pinRefArrayRemove_pushButton")

        self.pinRefArray_verticalLayout_2.addWidget(self.pinRefArrayRemove_pushButton)

        self.pinRefArray_verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.pinRefArray_verticalLayout_2.addItem(self.pinRefArray_verticalSpacer)

        self.pinRefArray_horizontalLayout.addLayout(self.pinRefArray_verticalLayout_2)

        self.gridLayout_4.addLayout(self.pinRefArray_horizontalLayout, 0, 0, 1, 1)

        self.gridLayout.addWidget(self.pinRefArray_groupBox, 3, 0, 1, 1)

        self.retranslateUi(Form)
        self.ikfk_slider.sliderMoved.connect(self.ikfk_spinBox.setValue)
        self.ikfk_spinBox.valueChanged.connect(self.ikfk_slider.setValue)

        QMetaObject.connectSlotsByName(Form)

    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", "Form", None))
        self.groupBox.setTitle("")
        self.ikfk_label.setText(QCoreApplication.translate("Form", "FK/IK Blend", None))
        self.maxStretch_label.setText(QCoreApplication.translate("Form", "Max Stretch", None))
        self.divisions_label.setText(QCoreApplication.translate("Form", "Divisions", None))
        self.leafJoints_checkBox.setText(QCoreApplication.translate("Form", "Leaf Joints", None))
        self.ikTR_checkBox.setText(
            QCoreApplication.translate("Form", "IK separated Trans and Rot ctl", None)
        )
        # if QT_CONFIG(tooltip)
        self.mirrorIK_checkBox.setToolTip(
            QCoreApplication.translate(
                "Form",
                "This option set the axis of the mid CTL (elbow) and the up vector control to move in a mirror behaviour ",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(statustip)
        self.mirrorIK_checkBox.setStatusTip(
            QCoreApplication.translate(
                "Form",
                "This option set the axis of the mid CTL (elbow) and the up vector control to move in a mirror behaviour ",
                None,
            )
        )
        # endif // QT_CONFIG(statustip)
        # if QT_CONFIG(whatsthis)
        self.mirrorIK_checkBox.setWhatsThis(
            QCoreApplication.translate(
                "Form",
                "This option set the axis of the mid CTL (elbow) and the up vector control to move in a mirror behaviour ",
                None,
            )
        )
        # endif // QT_CONFIG(whatsthis)
        self.mirrorIK_checkBox.setText(
            QCoreApplication.translate("Form", "Mirror IK Ctl  axis behaviour", None)
        )
        # if QT_CONFIG(tooltip)
        self.mirrorMid_checkBox.setToolTip(
            QCoreApplication.translate(
                "Form",
                "This option set the axis of the mid CTL (elbow) and the up vector control to move in a mirror behaviour ",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(statustip)
        self.mirrorMid_checkBox.setStatusTip(
            QCoreApplication.translate(
                "Form",
                "This option set the axis of the mid CTL (elbow) and the up vector control to move in a mirror behaviour ",
                None,
            )
        )
        # endif // QT_CONFIG(statustip)
        # if QT_CONFIG(whatsthis)
        self.mirrorMid_checkBox.setWhatsThis(
            QCoreApplication.translate(
                "Form",
                "This option set the axis of the mid CTL (elbow) and the up vector control to move in a mirror behaviour ",
                None,
            )
        )
        # endif // QT_CONFIG(whatsthis)
        self.mirrorMid_checkBox.setText(
            QCoreApplication.translate("Form", "Mirror Mid Ctl and UPV  axis behaviour", None)
        )
        # if QT_CONFIG(tooltip)
        self.useBlade_checkBox.setToolTip(
            QCoreApplication.translate(
                "Form",
                "<html><head/><body><p>If checked, will use a blade to control the wrist joint orientation. This doesn't affect the controls that are align with the arm plane.</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(statustip)
        self.useBlade_checkBox.setStatusTip("")
        # endif // QT_CONFIG(statustip)
        # if QT_CONFIG(whatsthis)
        self.useBlade_checkBox.setWhatsThis("")
        # endif // QT_CONFIG(whatsthis)
        self.useBlade_checkBox.setText(
            QCoreApplication.translate("Form", "Use Wrist Blade to orient wrist joint", None)
        )
        # if QT_CONFIG(tooltip)
        self.TPoseRest_checkBox.setToolTip(
            QCoreApplication.translate(
                "Form",
                "<html><head/><body><p>If checked, the Rest pose for controls will be in T Pose</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(statustip)
        self.TPoseRest_checkBox.setStatusTip(
            QCoreApplication.translate(
                "Form",
                "This option set the axis of the mid CTL (elbow) and the up vector control to move in a mirror behaviour ",
                None,
            )
        )
        # endif // QT_CONFIG(statustip)
        # if QT_CONFIG(whatsthis)
        self.TPoseRest_checkBox.setWhatsThis(
            QCoreApplication.translate(
                "Form",
                "This option set the axis of the mid CTL (elbow) and the up vector control to move in a mirror behaviour ",
                None,
            )
        )
        # endif // QT_CONFIG(whatsthis)
        self.TPoseRest_checkBox.setText(QCoreApplication.translate("Form", "Rest T Pose", None))
        self.squashStretchProfile_pushButton.setText(
            QCoreApplication.translate("Form", "Squash and Stretch Profile", None)
        )
        self.ikRefArray_groupBox.setTitle(
            QCoreApplication.translate("Form", "IK Reference Array", None)
        )
        self.ikRefArray_copyRef_pushButton.setText(
            QCoreApplication.translate("Form", "Copy from UpV Ref", None)
        )
        self.ikRefArrayAdd_pushButton.setText(QCoreApplication.translate("Form", "<<", None))
        self.ikRefArrayRemove_pushButton.setText(QCoreApplication.translate("Form", ">>", None))
        self.upvRefArray_groupBox.setTitle(
            QCoreApplication.translate("Form", "UpV Reference Array", None)
        )
        self.upvRefArray_copyRef_pushButton.setText(
            QCoreApplication.translate("Form", "Copy from IK Ref", None)
        )
        self.upvRefArrayAdd_pushButton.setText(QCoreApplication.translate("Form", "<<", None))
        self.upvRefArrayRemove_pushButton.setText(QCoreApplication.translate("Form", ">>", None))
        self.pinRefArray_groupBox.setTitle(
            QCoreApplication.translate("Form", "Pin Elbow Reference Array", None)
        )
        self.pinRefArray_copyRef_pushButton.setText(
            QCoreApplication.translate("Form", "Copy from IK Ref", None)
        )
        self.pinRefArrayAdd_pushButton.setText(QCoreApplication.translate("Form", "<<", None))
        self.pinRefArrayRemove_pushButton.setText(QCoreApplication.translate("Form", ">>", None))

    # retranslateUi
