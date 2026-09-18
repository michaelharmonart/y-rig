from maya.OpenMayaUI import MQtUtil
from Qt.QtCompat import wrapInstance
from Qt.QtWidgets import QWidget


def maya_main_window() -> QWidget:
    pointer = MQtUtil.mainWindow()
    return wrapInstance(int(pointer), QWidget)  # type: ignore
