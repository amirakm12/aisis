"""
Voice Processing Module for AI-ARTWORK
Provides unified interface for speech recognition and text-to-speech
"""

from .whisper_asr import WhisperASR, VoiceActivityDetector, whisper_asr, vad
from .bark_tts import BarkTTS, bark_tts

__all__ = [
    "WhisperASR",
    "VoiceActivityDetector", 
    "BarkTTS",
    "whisper_asr",
    "vad",
    "bark_tts"
]
