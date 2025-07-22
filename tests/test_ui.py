import pytest

# Assuming pytest-qt is installed
def test_main_window(qtbot):
    from src.ui.modern_interface import ModernMainWindow  # adjust if needed
    window = ModernMainWindow()
    qtbot.addWidget(window)
    assert window.isVisible()
