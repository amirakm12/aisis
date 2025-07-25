from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QTimer
from PySide6.QtCore import Qt

class Notification(QWidget):
    def __init__(self, message, duration=3000, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.ToolTip)
        layout = QVBoxLayout(self)
        label = QLabel(message)
        layout.addWidget(label)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        QTimer.singleShot(duration, self.close)
        self.show() 