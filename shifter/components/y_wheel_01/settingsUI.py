################################################################################
## Form generated from reading UI file 'settingsUIrsieea.ui'
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
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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


class Ui_Form:
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName("Form")
        Form.resize(255, 290)
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
        self.mode_label = QLabel(self.groupBox)
        self.mode_label.setObjectName("mode_label")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.mode_label)

        self.mode_comboBox = QComboBox(self.groupBox)
        self.mode_comboBox.addItem("")
        self.mode_comboBox.addItem("")
        self.mode_comboBox.addItem("")
        self.mode_comboBox.setObjectName("mode_comboBox")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mode_comboBox.sizePolicy().hasHeightForWidth())
        self.mode_comboBox.setSizePolicy(sizePolicy)

        self.formLayout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.mode_comboBox)

        self.ikfk_label = QLabel(self.groupBox)
        self.ikfk_label.setObjectName("ikfk_label")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.ikfk_label)

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

        self.formLayout.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_3)

        # ---- Wheel Type ----
        self.wheelType_label = QLabel(self.groupBox)
        self.wheelType_label.setObjectName("wheelType_label")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.wheelType_label)

        self.wheelType_comboBox = QComboBox(self.groupBox)
        self.wheelType_comboBox.setObjectName("wheelType_comboBox")
        self.wheelType_comboBox.addItem("")  # Front
        self.wheelType_comboBox.addItem("")  # Rear

        self.formLayout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.wheelType_comboBox)

        self.verticalLayout.addLayout(self.formLayout)

        self.neutralPose_checkBox = QCheckBox(self.groupBox)
        self.neutralPose_checkBox.setObjectName("neutralPose_checkBox")

        self.verticalLayout.addWidget(self.neutralPose_checkBox)

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

        self.retranslateUi(Form)
        self.ikfk_slider.sliderMoved.connect(self.ikfk_spinBox.setValue)
        self.ikfk_spinBox.valueChanged.connect(self.ikfk_slider.setValue)

        QMetaObject.connectSlotsByName(Form)

    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", "Form", None))
        self.groupBox.setTitle("")
        self.mode_label.setText(QCoreApplication.translate("Form", "Mode:", None))
        self.mode_comboBox.setItemText(0, QCoreApplication.translate("Form", "FK", None))
        self.mode_comboBox.setItemText(1, QCoreApplication.translate("Form", "IK", None))
        self.mode_comboBox.setItemText(2, QCoreApplication.translate("Form", "FK/IK", None))

        self.ikfk_label.setText(QCoreApplication.translate("Form", "IK/FK Blend:", None))
        self.neutralPose_checkBox.setText(QCoreApplication.translate("Form", "Nuetral pose", None))
        self.ikRefArray_groupBox.setTitle(
            QCoreApplication.translate("Form", "IK Reference Array", None)
        )
        self.ikRefArrayAdd_pushButton.setText(QCoreApplication.translate("Form", "<<", None))
        self.ikRefArrayRemove_pushButton.setText(QCoreApplication.translate("Form", ">>", None))

        self.wheelType_label.setText(QCoreApplication.translate("Form", "Wheel Type:", None))
        self.wheelType_comboBox.setItemText(0, QCoreApplication.translate("Form", "Front", None))
        self.wheelType_comboBox.setItemText(1, QCoreApplication.translate("Form", "Rear", None))

    # retranslateUi
