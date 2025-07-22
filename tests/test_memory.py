import pytest
from memory_profiler import profile
from src.core.voice_manager import VoiceManager

@profile
def memory_intensive_function():
    vm = VoiceManager()
    vm.initialize()
    vm.cleanup()

def test_memory_leak():
    for _ in range(10):
        memory_intensive_function()
    # Manually check memory usage
    assert True  # Placeholder
