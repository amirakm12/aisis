"""
UI component tests for AISIS
Tests the Qt-based user interface components and interactions
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow


@pytest.fixture
def app(qtbot):
    """Qt application fixture"""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def main_window(app, qtbot):
    """Main window fixture"""
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_main_window_creation(main_window):
    """Test main window initialization"""
    assert main_window.windowTitle() == "AISIS - AI Creative Studio"
    assert main_window.size().width() >= 1280
    assert main_window.size().height() >= 720


def test_voice_control_button(main_window, qtbot):
    """Test voice control button functionality"""
    button = main_window.voice_button
    assert not button.isChecked()

    # Click voice control button
    qtbot.mouseClick(button, Qt.LeftButton)
    assert button.isChecked()
    assert main_window.voice_active

    # Click again to deactivate
    qtbot.mouseClick(button, Qt.LeftButton)
    assert not button.isChecked()
    assert not main_window.voice_active


def test_toolbar_actions(main_window, qtbot):
    """Test toolbar actions"""
    toolbar = main_window.findChild(QToolBar)
    assert toolbar is not None

    # Test each action
    actions = toolbar.actions()
    assert len(actions) >= 5  # New, Open, Save, Undo, Redo

    # Verify action names
    action_names = [a.text() for a in actions]
    assert "New Project" in action_names
    assert "Open Project" in action_names
    assert "Save Project" in action_names
    assert "Undo" in action_names
    assert "Redo" in action_names


def test_status_bar(main_window):
    """Test status bar components"""
    status_bar = main_window.statusBar()
    assert status_bar is not None

    # Check voice status label
    voice_status = main_window.voice_status
    assert voice_status is not None
    assert "Voice Control: Inactive" in voice_status.text()


def test_theme_settings(main_window):
    """Test theme application"""
    palette = main_window.palette()

    # Check dark theme colors
    if main_window.config.ui.theme == "dark":
        assert palette.color(palette.Window).name() == "#353535"
        assert palette.color(palette.WindowText) == Qt.white
        assert palette.color(palette.Base).name() == "#191919"


def test_window_close(main_window, qtbot):
    """Test window closure and cleanup"""
    with qtbot.waitSignal(main_window.voice_status_changed, timeout=1000):
        # Activate voice control
        main_window.voice_button.click()

    # Close window
    main_window.close()

    # Verify voice control is deactivated
    assert not main_window.voice_active


def test_keyboard_shortcuts(main_window, qtbot):
    """Test keyboard shortcuts"""
    # Test Ctrl+N for new project
    with qtbot.waitSignal(main_window.statusBar().messageChanged, timeout=1000):
        QTest.keySequence(main_window, "Ctrl+N")

    # Test Ctrl+S for save
    with qtbot.waitSignal(main_window.statusBar().messageChanged, timeout=1000):
        QTest.keySequence(main_window, "Ctrl+S")

    # Test Ctrl+Z for undo
    with qtbot.waitSignal(main_window.statusBar().messageChanged, timeout=1000):
        QTest.keySequence(main_window, "Ctrl+Z")


def test_voice_command_handling(main_window, qtbot):
    """Test voice command processing"""
    test_command = "make it brighter"

    # Simulate voice command
    with qtbot.waitSignal(main_window.statusBar().messageChanged, timeout=1000):
        main_window._handle_voice_command(test_command)

    # Verify status bar message
    assert test_command in main_window.statusBar().currentMessage()


def test_window_responsiveness(main_window, qtbot):
    """Test window responsiveness to resizing"""
    original_size = main_window.size()

    # Test window resizing
    new_size = original_size * 1.5
    main_window.resize(new_size)
    qtbot.wait(100)

    # Verify minimum size constraints
    assert main_window.size().width() >= 1280
    assert main_window.size().height() >= 720


if __name__ == "__main__":
    pytest.main([__file__])
