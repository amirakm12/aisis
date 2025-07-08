"""
UI component tests for AISIS
                         Tests the Qt-based user interface components and interactions
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.agent_explain_dialog import AgentExplainDialog
from src.ui.learning_panel import LearningPanel
from src.ui.model_zoo_dialog import ModelZooDialog
from src.ui.settings_panel import SettingsDialog
from src.ui.loading_screen import LoadingScreen
from src.ui.onboarding_dialog import OnboardingDialog
from src.ui.notifications import Notification
from src.ui.plugin_manager_dialog import PluginManagerDialog
from src.ui.tour_dialog import TourDialog
from src.ui.crash_reporting_dialog import CrashReportingDialog

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

def test_agent_explain_dialog_shows_doc(qtbot, app):
    class DummyAgent:
        __doc__ = "Test agent doc"
    dialog = AgentExplainDialog(agent_registry={"Dummy": DummyAgent()})
    qtbot.addWidget(dialog)
    dialog.show()
    assert "Test agent doc" in dialog.explanation.toPlainText()

def test_learning_panel(qtbot, app):
    panel = LearningPanel()
    qtbot.addWidget(panel)
    panel.show()
    assert panel.status_label.text() == "Learning Status: Idle"

def test_model_zoo_dialog(qtbot, app):
    class DummyModelZoo:
        def list_models(self):
            return [{"name": "ModelA", "type": "vision", "version": "1.0", "status": "available"}]
    dialog = ModelZooDialog(DummyModelZoo())
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.model_list.count() == 1

def test_settings_panel(qtbot, app):
    config = {"ui.theme": "dark", "paths.models_dir": "models", "gpu.use_cuda": True}
    dialog = SettingsDialog(config)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.theme_box.text() == "dark"

def test_loading_screen(qtbot, app):
    screen = LoadingScreen("Loading...")
    qtbot.addWidget(screen)
    screen.show()
    assert screen.label.text() == "Loading..."

def test_onboarding_dialog(qtbot, app):
    dialog = OnboardingDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.windowTitle() == "Welcome to AISIS!"

def test_notification(qtbot, app):
    notif = Notification("Test notification", duration=100, parent=None)
    qtbot.addWidget(notif)
    notif.show()
    assert notif.isVisible()

def test_plugin_manager_dialog(qtbot, app):
    class DummyPluginManager:
        pass
    dialog = PluginManagerDialog(DummyPluginManager())
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.windowTitle() == "Plugin Manager"

def test_tour_dialog(qtbot, app):
    dialog = TourDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.windowTitle() == "Welcome Tour"

def test_crash_reporting_dialog(qtbot, app):
    config = {"crash_reporting.opt_in": False}
    dialog = CrashReportingDialog(config)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.windowTitle() == "Crash Reporting"

def test_plugin_sandbox_success():
                             assert run_plugin_in_sandbox(DummyPlugin) == "ok"

@pytest.mark.asyncio
async def test_agent_process_error():
    agent = DummyAgent()
    with pytest.raises(ValueError):
        await agent._process({"fail": True})

def test_plugin_agent_integration():
    class AgentPlugin:
        def run(self):
            class DummyAgent:
                def process(self):
                    return "agent result"
            agent = DummyAgent()
            return agent.process()
    result = run_plugin_in_sandbox(AgentPlugin)
    assert result == "agent result"

@pytest.mark.asyncio
@pytest.mark.parametrize("n", [1, 10, 50])
async def test_agent_stress(n):
    agent = DummyAgent()
    results = []
    for _ in range(n):
        result = await agent._process({})
        results.append(result)
    assert all(r["status"] == "success" for r in results)

if __name__ == "__main__":
    pytest.main([__file__])
                         