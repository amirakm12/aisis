import pytest
from src.voice import VoiceManager  # Adjust import

@pytest.mark.asyncio
async def test_voice_full_flow():
    vm = VoiceManager()
    audio = 'test_audio.wav'
    text = await vm.asr(audio)
    assert text == 'expected'
    response = 'response text'
    await vm.tts(response)
    # Assert audio generated