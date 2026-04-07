# -*- coding: utf-8 -*-

from Qt.QtCore import QCoreApplication, QMetaObject
from Qt.QtWidgets import (
    QGridLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QSizePolicy,
    QSpacerItem,
)


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName("Form")

        Form.resize(300, 200)

        # ===== MAIN GRID =====
        self.gridLayout = QGridLayout(Form)

        # ===== FORM LAYOUT =====
        self.formLayout = QFormLayout()

        # ---- Name ----
        self.name_label = QLabel(Form)
        self.name_label.setText("Name")
        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.name_label)

        self.name_lineEdit = QLineEdit(Form)
        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.name_lineEdit)

        # ---- Side ----
        self.side_label = QLabel(Form)
        self.side_label.setText("Side")
        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.side_label)

        self.side_comboBox = QComboBox(Form)
        self.side_comboBox.addItems(["Center", "Left", "Right"])
        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.side_comboBox)

        # ---- Wheels ----
        self.wheels_label = QLabel(Form)
        self.wheels_label.setText("Wheels Connector")
        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.wheels_label)

        self.wheels_lineEdit = QLineEdit(Form)
        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.wheels_lineEdit)

        # Add form to grid
        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)

        # Spacer (push UI up like your friend's)
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.gridLayout.addItem(self.verticalSpacer, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", "Car Body Settings", None))
