from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QCheckBox, QPushButton
from src.ui.notifications import Notification
from src.ui.plugin_manager_dialog import PluginManagerDialog
from src.plugins.plugin_manager import PluginManager
from src.ui.theme_manager import ThemeManager
from PySide6.QtWidgets import QApplication
from src.ui.tour_dialog import TourDialog
from src.core.config_validation import AISISConfig, ValidationError
import json
from src.core.logging_setup import setup_logging
from src.plugins.sandbox import run_plugin_in_sandbox
import asyncio
import websockets

class CrashReportingDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crash Reporting")
        layout = QVBoxLayout(self)
        self.label = QLabel("Help us improve AISIS by sending anonymous crash reports.")
        self.opt_in_checkbox = QCheckBox("Enable anonymous crash reporting")
        self.opt_in_checkbox.setChecked(config.get("crash_reporting.opt_in", False))
        self.save_btn = QPushButton("Save")
        layout.addWidget(self.label)
        layout.addWidget(self.opt_in_checkbox)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)
        self.save_btn.clicked.connect(self.save_opt_in)
        self.config = config

    def save_opt_in(self):
        self.config["crash_reporting.opt_in"] = self.opt_in_checkbox.isChecked()
        self.accept()

    def on_some_event(self):
        Notification("Operation completed successfully!", duration=2000, parent=self)

    def on_save_success(self):
        Notification("Image saved successfully!", parent=self)

    def open_plugin_manager(self):
        dlg = PluginManagerDialog(PluginManager(), self)
        dlg.exec()

    def enable_selected_plugin(self):
        selected = self.plugin_list.currentItem()
        if selected:
            self.plugin_manager.enable_plugin(selected.text())

    def on_theme_change(self, theme):
        if theme == "dark":
            ThemeManager.apply_dark(QApplication.instance())
        else:
            ThemeManager.apply_light(QApplication.instance())

    def show_tour(self):
        dlg = TourDialog(self)
        dlg.exec()

    def import_settings(self, path):
        with open(path, "r") as f:
            imported = json.load(f)
        try:
            validated = AISISConfig(**imported)
            self.config = validated.dict()
            # Save and reload app config as needed
        except ValidationError as e:
            Notification(f"Invalid settings: {e}", parent=self)

    def handle_crash(exc):
        if config.get("crash_reporting.opt_in", False):
            send_crash_report(exc)
        else:
            logger.error(str(exc))

    def run_plugin(self, plugin_class, *args, **kwargs):
        try:
            result = run_plugin_in_sandbox(plugin_class, *args, **kwargs)
        except Exception as e:
            Notification(f"Plugin error: {e}", parent=self)

    async def send_message(self, message):
        async with websockets.connect("ws://localhost:8765") as ws:
            await ws.send(message)
            response = await ws.recv()
            self.chat_history.addItem(response)

setup_logging(log_level="INFO") 

def test_agent_explain_dialog(qtbot):
    from src.ui.agent_explain_dialog import AgentExplainDialog
    dialog = AgentExplainDialog(agent_registry={"TestAgent": object()})
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.isVisible() 

def test_plugin_sandbox():
    from src.plugins.sandbox import run_plugin_in_sandbox
    class DummyPlugin:
        def run(self): return "ok"
    assert run_plugin_in_sandbox(DummyPlugin) == "ok" 

def test_agent_explain_dialog_shows_doc(qtbot):
    class DummyAgent:
        __doc__ = "Test agent doc"
    dialog = AgentExplainDialog(agent_registry={"Dummy": DummyAgent()})
    qtbot.addWidget(dialog)
    dialog.show()
    assert "Test agent doc" in dialog.explanation.toPlainText() 

def test_plugin_manager_loads_plugins():
    from src.plugins.plugin_manager import PluginManager
    pm = PluginManager()
    pm.load_plugins()
    assert isinstance(pm.plugins, dict) 

import pytest
@pytest.mark.parametrize("size", [(512, 512), (1024, 1024)])
def test_agent_performance_on_large_images(size):
    # Simulate agent processing on large images
    pass 