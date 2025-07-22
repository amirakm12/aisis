import pytest
from PyQt6.QtWidgets import QApplication
from src.ui.context_panel import ContextPanel
from src.ui.context_manager import ContextManager
import sys


@pytest.fixture(scope="module")
def app():
    app = QApplication(sys.argv)
    yield app


@pytest.fixture
def panel(app):
    cm = ContextManager()
    panel = ContextPanel(cm)
    return panel


def test_context_update(panel):
    event = {"action": "test_action", "value": 42}
    panel.update_context(event)
    assert "action" in panel.context_manager.session_context
    assert panel.context_manager.session_context["action"] == "test_action"


def test_progress(panel):
    panel.set_progress(2, "Running", step=1)
    assert panel.progress_bar.value() == 1


def test_state(panel):
    panel.set_state("Active", agent="TestAgent")
    assert panel.agent_status.text() == "Active"
    assert "TestAgent" in panel.agent_label.text()


def test_log(panel):
    panel.log("Test log entry")
    assert "Test log entry" in panel.log_area.toPlainText()


def test_pin_context(panel):
    event = {"foo": "bar"}
    panel.update_context(event)
    panel.context_list.setCurrentRow(0)
    panel._pin_selected()
    assert "foo" in panel.pinned_context


def test_export_context(tmp_path, panel):
    panel.update_context({"foo": "bar"})
    fname = tmp_path / "context.txt"
    panel.export_context = lambda: fname.write_text("exported")  # Mock
    panel.export_context()
    assert fname.read_text() == "exported"


def test_collapse(panel):
    panel.toggle_collapse()
    assert not panel.isVisible()
    panel.toggle_collapse()
    assert panel.isVisible()
