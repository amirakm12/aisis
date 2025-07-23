import pytest
from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

@pytest.fixture
def app(qtbot):
    app = QApplication([])
    window = MainWindow()
    qtbot.addWidget(window)
    return window

def test_ui_button_click(app, qtbot):
    # Simulate button click and assert
    pass