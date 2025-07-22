import pytest

from src.core.voice_manager import VoiceManager

@pytest.fixture
def voice_manager():
    vm = VoiceManager()
    yield vm
    vm.cleanup()

@pytest.mark.asyncio
async def test_initialize(voice_manager):
    await voice_manager.initialize()
    assert voice_manager.initialized
    assert voice_manager.asr is not None
    assert voice_manager.tts is not None

@pytest.mark.asyncio
async def test_synthesize(voice_manager):
    await voice_manager.initialize()
    output = await voice_manager.synthesize("Test speech synthesis")
    assert output.exists()
    assert output.suffix == ".wav"
