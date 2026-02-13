# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'settingsUIXJxXqO.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from Qt.QtCore import (
    QCoreApplication,
    QMetaObject,
    QSize,
    Qt,
)
from Qt.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
)


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName("Form")
        Form.resize(339, 332)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.formLayout = QFormLayout(self.groupBox)
        self.formLayout.setObjectName("formLayout")
        self.preserve_length_label = QLabel(self.groupBox)
        self.preserve_length_label.setObjectName("preserve_length_label")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.preserve_length_label)

        self.preserve_length_layout = QHBoxLayout()
        self.preserve_length_layout.setObjectName("preserve_length_layout")
        self.preserve_length_slider = QSlider(self.groupBox)
        self.preserve_length_slider.setObjectName("preserve_length_slider")
        self.preserve_length_slider.setMinimumSize(QSize(0, 15))
        self.preserve_length_slider.setMaximum(100)
        self.preserve_length_slider.setSliderPosition(100)
        self.preserve_length_slider.setOrientation(Qt.Orientation.Horizontal)

        self.preserve_length_layout.addWidget(self.preserve_length_slider)

        self.preserve_length_spinBox = QSpinBox(self.groupBox)
        self.preserve_length_spinBox.setObjectName("preserve_length_spinBox")
        self.preserve_length_spinBox.setMaximum(100)

        self.preserve_length_layout.addWidget(self.preserve_length_spinBox)

        self.formLayout.setLayout(0, QFormLayout.ItemRole.FieldRole, self.preserve_length_layout)

        self.divisions_label = QLabel(self.groupBox)
        self.divisions_label.setObjectName("divisions_label")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.divisions_label)

        self.division_spinBox = QSpinBox(self.groupBox)
        self.division_spinBox.setObjectName("division_spinBox")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.division_spinBox.sizePolicy().hasHeightForWidth())
        self.division_spinBox.setSizePolicy(sizePolicy)
        self.division_spinBox.setMinimum(2)
        self.division_spinBox.setMaximum(99)
        self.division_spinBox.setValue(4)

        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.division_spinBox)

        self.ctl_world_orient_label = QLabel(self.groupBox)
        self.ctl_world_orient_label.setObjectName("ctl_world_orient_label")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.ctl_world_orient_label)

        self.ctl_world_orient_checkBox = QCheckBox(self.groupBox)
        self.ctl_world_orient_checkBox.setObjectName("ctl_world_orient_checkBox")
        self.ctl_world_orient_checkBox.setEnabled(True)

        self.formLayout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.ctl_world_orient_checkBox)

        self.leafJoints_label = QLabel(self.groupBox)
        self.leafJoints_label.setObjectName("leafJoints_label")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.leafJoints_label)

        self.leafJoints_checkBox = QCheckBox(self.groupBox)
        self.leafJoints_checkBox.setObjectName("leafJoints_checkBox")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.FieldRole, self.leafJoints_checkBox)

        self.squashStretchProfile_pushButton = QPushButton(self.groupBox)
        self.squashStretchProfile_pushButton.setObjectName("squashStretchProfile_pushButton")

        self.formLayout.setWidget(
            5, QFormLayout.ItemRole.SpanningRole, self.squashStretchProfile_pushButton
        )

        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.gridLayout.addItem(self.verticalSpacer, 1, 0, 1, 1)

        self.retranslateUi(Form)
        self.preserve_length_slider.valueChanged.connect(self.preserve_length_spinBox.setValue)
        self.preserve_length_spinBox.valueChanged.connect(self.preserve_length_slider.setValue)

        QMetaObject.connectSlotsByName(Form)

    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", "Form", None))
        self.groupBox.setTitle("")
        self.preserve_length_label.setText(
            QCoreApplication.translate("Form", "Preserve Length", None)
        )
        self.divisions_label.setText(QCoreApplication.translate("Form", "Divisions", None))
        self.ctl_world_orient_label.setText(
            QCoreApplication.translate("Form", "CTL World Orient", None)
        )
        # if QT_CONFIG(tooltip)
        self.ctl_world_orient_checkBox.setToolTip(
            QCoreApplication.translate(
                "Form",
                "<html><head/><body><p>If checked the IK controls will be oriented to world space in XYZ</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.ctl_world_orient_checkBox.setText("")
        self.leafJoints_label.setText(QCoreApplication.translate("Form", "Leaf Joints", None))
        self.leafJoints_checkBox.setText("")
        self.squashStretchProfile_pushButton.setText(
            QCoreApplication.translate("Form", "Squash and Stretch Profile", None)
        )

    # retranslateUi
