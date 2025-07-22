from PySide6.QtWidgets import QSplashScreen, QLabel
from PySide6.QtCore import Qt


class LoadingScreen(QSplashScreen):
    def __init__(self, message="Loading..."):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.label = QLabel(message, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.showMessage(message, Qt.AlignBottom | Qt.AlignCenter, Qt.white)
