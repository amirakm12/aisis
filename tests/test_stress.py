import pytest
from src.core.voice_manager import VoiceManager

@pytest.mark.asyncio
async def test_stress_initialize():
    for _ in range(10):
        vm = VoiceManager()
        await vm.initialize()
        vm.cleanup()
