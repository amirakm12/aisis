from src.plugins.sandbox import run_plugin_in_sandbox

class DummyPlugin:
    def run(self):
        return "ok"

class ErrorPlugin:
    def run(self):
        raise ValueError("fail")

def test_plugin_sandbox_success():
    assert run_plugin_in_sandbox(DummyPlugin) == "ok"

def test_plugin_sandbox_error():
    try:
        run_plugin_in_sandbox(ErrorPlugin)
    except RuntimeError as e:
        assert "Plugin error" in str(e)
    else:
        assert False, "Expected RuntimeError" 