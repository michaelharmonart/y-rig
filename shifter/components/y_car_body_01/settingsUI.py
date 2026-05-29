from Qt.QtCore import QCoreApplication, QMetaObject
from Qt.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpacerItem,
)


class Ui_Form:
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
        self.name_lineEdit = QLineEdit(Form)
        self.formLayout.addRow(self.name_label, self.name_lineEdit)

        # ---- Side ----
        self.side_label = QLabel(Form)
        self.side_label.setText("Side")
        self.side_comboBox = QComboBox(Form)
        self.side_comboBox.addItems(["Center", "Left", "Right"])
        self.formLayout.addRow(self.side_label, self.side_comboBox)

        # ---- Wheels ----
        self.wheels_label = QLabel(Form)
        self.wheels_label.setText("Wheels Connector")
        self.wheels_lineEdit = QLineEdit(Form)
        self.formLayout.addRow(self.wheels_label, self.wheels_lineEdit)

        # ---- Wheel Radius ----
        self.wheelRadius_label = QLabel(Form)
        self.wheelRadius_label.setText("Wheel Radius")
        self.wheelRadius_lineEdit = QLineEdit(Form)
        self.formLayout.addRow(self.wheelRadius_label, self.wheelRadius_lineEdit)

        # ---- Wheel Radius 2 ----
        self.wheelRadius2_label = QLabel(Form)
        self.wheelRadius2_label.setText("Wheel Radius 2")
        self.wheelRadius2_lineEdit = QLineEdit(Form)
        self.formLayout.addRow(self.wheelRadius2_label, self.wheelRadius2_lineEdit)

        # Add form to grid
        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)

        # Spacer (push UI up like your friend's)
        self.verticalSpacer = QSpacerItem(
            20,
            40,
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Expanding,
        )
        self.gridLayout.addItem(self.verticalSpacer, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", "Car Body Settings", None))
