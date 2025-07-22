"""
Core functionality tests for AISIS
"""

import asyncio
import pytest
import torch
from pathlib import Path

from src.core.config import config
from src.core.gpu_utils import gpu_manager, model_loader
from src.core.voice_manager import voice_manager
from src.core.voice import whisper_asr, bark_tts


@pytest.fixture
async def initialized_voice_manager():
    """Fixture for initialized voice manager"""
    await voice_manager.initialize()
    yield voice_manager
    voice_manager.cleanup()


def test_config_loading():
    """Test configuration system"""
    assert config is not None
    assert config.project_root is not None
    assert isinstance(config.gpu.device, str)
    assert isinstance(config.voice.sample_rate, int)


def test_gpu_manager():
    """Test GPU management system"""
    assert gpu_manager is not None
    assert isinstance(gpu_manager.device, torch.device)
    assert isinstance(gpu_manager.gpu_info, dict)

    # Test GPU utilities
    gpu_manager.clear_cache()
    memory_info = gpu_manager.get_memory_info()
    assert isinstance(memory_info, dict)


@pytest.mark.asyncio
async def test_whisper_asr():
    """Test Whisper ASR initialization"""
    await whisper_asr.initialize()
    model_info = whisper_asr.get_model_info()

    assert model_info["status"] == "loaded"
    assert model_info["model_size"] == config.voice.whisper_model
    assert model_info["language"] == "en"


@pytest.mark.asyncio
async def test_bark_tts():
    """Test Bark TTS initialization"""
    await bark_tts.initialize()

    # Test voice presets
    presets = bark_tts.get_available_voices()
    assert isinstance(presets, dict)
    assert "default" in presets

    # Test current voice
    current_voice = bark_tts.get_current_voice()
    assert isinstance(current_voice, dict)
    assert "name" in current_voice


@pytest.mark.asyncio
async def test_voice_manager(initialized_voice_manager):
    """Test voice manager functionality"""
    status = initialized_voice_manager.get_status()

    assert isinstance(status, dict)
    assert "is_listening" in status
    assert "is_speaking" in status
    assert "asr_info" in status
    assert "tts_voice" in status


def test_model_loader():
    """Test model loading utilities"""
    assert model_loader is not None

    # Test optimal batch size calculation
    batch_size = model_loader.gpu_manager.get_optimal_batch_size(1000)
    assert isinstance(batch_size, int)
    assert batch_size >= 1


@pytest.mark.asyncio
async def test_basic_voice_interaction(initialized_voice_manager):
    """Test basic voice interaction flow"""
    # Test text-to-speech
    test_text = "Hello, this is a test."
    audio_data = await bark_tts.generate_speech(test_text)
    assert isinstance(audio_data, numpy.ndarray)
    assert len(audio_data) > 0

    # Test voice activity detection
    from src.core.voice import vad

    assert vad.is_speech(audio_data)


if __name__ == "__main__":
    pytest.main([__file__])
