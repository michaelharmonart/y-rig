"""Settings UI for y_finger_01 component."""

from mgear.vendor.Qt import QtCore, QtWidgets  # type: ignore


class Ui_Form:
    def setupUi(self, Form):
        self.form = Form
        self.setup_ui_options(Form)
        self.create_controls(Form)
        self.create_layout(Form)
        self.create_connections()

    def setup_ui_options(self, Form):
        Form.setObjectName("Form")
        Form.resize(255, 290)

    def create_controls(self, Form):
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")

        self.mode_label = QtWidgets.QLabel("Mode:", self.groupBox)
        self.mode_label.setObjectName("mode_label")

        self.mode_comboBox = QtWidgets.QComboBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.mode_comboBox.setSizePolicy(sizePolicy)
        self.mode_comboBox.setObjectName("mode_comboBox")
        self.mode_comboBox.addItem("FK")
        self.mode_comboBox.addItem("IK")
        self.mode_comboBox.addItem("FK/IK")

        self.ikfk_label = QtWidgets.QLabel("IK/FK Blend:", self.groupBox)
        self.ikfk_label.setObjectName("ikfk_label")

        self.ikfk_slider = QtWidgets.QSlider(self.groupBox)
        self.ikfk_slider.setMinimumSize(QtCore.QSize(0, 15))
        self.ikfk_slider.setMaximum(100)
        self.ikfk_slider.setOrientation(QtCore.Qt.Horizontal)
        self.ikfk_slider.setObjectName("ikfk_slider")

        self.ikfk_spinBox = QtWidgets.QSpinBox(self.groupBox)
        self.ikfk_spinBox.setMaximum(100)
        self.ikfk_spinBox.setObjectName("ikfk_spinBox")

        self.neutralPose_checkBox = QtWidgets.QCheckBox("Neutral pose", self.groupBox)
        self.neutralPose_checkBox.setObjectName("neutralPose_checkBox")

        self.ikRefArray_groupBox = QtWidgets.QGroupBox("IK Reference Array", Form)
        self.ikRefArray_groupBox.setObjectName("ikRefArray_groupBox")

        self.ikRefArray_listWidget = QtWidgets.QListWidget(self.ikRefArray_groupBox)
        self.ikRefArray_listWidget.setDragDropOverwriteMode(True)
        self.ikRefArray_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.ikRefArray_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.ikRefArray_listWidget.setAlternatingRowColors(True)
        self.ikRefArray_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.ikRefArray_listWidget.setSelectionRectVisible(False)
        self.ikRefArray_listWidget.setObjectName("ikRefArray_listWidget")

        self.ikRefArrayAdd_pushButton = QtWidgets.QPushButton("<<", self.ikRefArray_groupBox)
        self.ikRefArrayAdd_pushButton.setObjectName("ikRefArrayAdd_pushButton")

        self.ikRefArrayRemove_pushButton = QtWidgets.QPushButton(">>", self.ikRefArray_groupBox)
        self.ikRefArrayRemove_pushButton.setObjectName("ikRefArrayRemove_pushButton")

        self.isThumb_checkBox = QtWidgets.QCheckBox("Is Thumb", self.groupBox)
        self.isThumb_checkBox.setObjectName("isThumb_checkBox")

    def create_layout(self, Form):
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")

        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.mode_label)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.mode_comboBox)
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.ikfk_label)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_3.addWidget(self.ikfk_slider)
        self.horizontalLayout_3.addWidget(self.ikfk_spinBox)
        self.formLayout.setLayout(1, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_3)

        self.verticalLayout.addLayout(self.formLayout)
        self.verticalLayout.addWidget(self.neutralPose_checkBox)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        self.verticalLayout.addWidget(self.neutralPose_checkBox)
        self.verticalLayout.addWidget(self.isThumb_checkBox)

        self.gridLayout_3 = QtWidgets.QGridLayout(self.ikRefArray_groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.ikRefArray_horizontalLayout = QtWidgets.QHBoxLayout()
        self.ikRefArray_horizontalLayout.setObjectName("ikRefArray_horizontalLayout")

        self.ikRefArray_verticalLayout_1 = QtWidgets.QVBoxLayout()
        self.ikRefArray_verticalLayout_1.setObjectName("ikRefArray_verticalLayout_1")
        self.ikRefArray_verticalLayout_1.addWidget(self.ikRefArray_listWidget)

        self.ikRefArray_verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.ikRefArray_verticalLayout_2.setObjectName("ikRefArray_verticalLayout_2")
        self.ikRefArray_verticalLayout_2.addWidget(self.ikRefArrayAdd_pushButton)
        self.ikRefArray_verticalLayout_2.addWidget(self.ikRefArrayRemove_pushButton)
        self.ikRefArray_verticalLayout_2.addStretch(1)

        self.ikRefArray_horizontalLayout.addLayout(self.ikRefArray_verticalLayout_1)
        self.ikRefArray_horizontalLayout.addLayout(self.ikRefArray_verticalLayout_2)
        self.gridLayout_3.addLayout(self.ikRefArray_horizontalLayout, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.ikRefArray_groupBox, 1, 0, 1, 1)

    def create_connections(self):
        return
