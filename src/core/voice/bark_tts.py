"""
Bark Text-to-Speech Module
Handles high-quality voice synthesis using Bark TTS with GPU acceleration
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from loguru import logger

from ..config import config
from ..gpu_utils import gpu_manager


class BarkTTS:
    """GPU-accelerated Bark TTS for high-quality voice synthesis"""

    def __init__(self):
        self.device = gpu_manager.device
        self.sample_rate = SAMPLE_RATE
        self.is_initialized = False
        self.voice_presets = self._load_voice_presets()
        self.current_voice = None

        # Cache for generated audio to avoid regeneration of common phrases
        self.audio_cache: Dict[str, np.ndarray] = {}
        self.max_cache_size = 100  # Maximum number of cached audio samples

        logger.info("Initializing Bark TTS system")

    async def initialize(self) -> None:
        """Initialize Bark models and move to GPU"""
        if self.is_initialized:
            return

        try:
            logger.info("Loading Bark models...")

            # Preload models with GPU optimization
            with torch.cuda.device(gpu_manager.device):
                preload_models(
                    text_use_gpu=True,
                    coarse_use_gpu=True,
                    fine_use_gpu=True,
                    codec_use_gpu=True,
                )

            self.is_initialized = True
            logger.info("Bark models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Bark TTS: {e}")
            raise

    def _load_voice_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load available voice presets"""
        presets = {
            "default": {
                "name": "default",
                "description": "Default neutral voice",
                "voice_id": None,
            },
            "announcer": {
                "name": "announcer",
                "description": "Professional announcer voice",
                "voice_id": "v2/en_speaker_6",
            },
            "narrator": {
                "name": "narrator",
                "description": "Clear narrative voice",
                "voice_id": "v2/en_speaker_9",
            },
        }

        # Load custom voice presets if available
        custom_presets_path = Path("config/voice_presets.json")
        if custom_presets_path.exists():
            import json

            with open(custom_presets_path, "r") as f:
                custom_presets = json.load(f)
                presets.update(custom_presets)

        return presets

    def set_voice(self, voice_name: str) -> None:
        """Set the current voice preset"""
        if voice_name not in self.voice_presets:
            logger.warning(
                f"Voice preset '{voice_name}' not found. Using default."
            )
            voice_name = "default"

        self.current_voice = self.voice_presets[voice_name]
        logger.info(f"Set voice to: {voice_name}")

    def _get_cache_key(self, text: str, voice_name: str) -> str:
        """Generate cache key for text and voice combination"""
        return f"{voice_name}:{text}"

    def _cleanup_cache(self) -> None:
        """Remove oldest entries if cache exceeds maximum size"""
        if len(self.audio_cache) > self.max_cache_size:
            # Remove oldest 20% of entries
            num_to_remove = int(self.max_cache_size * 0.2)
            keys_to_remove = list(self.audio_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self.audio_cache[key]

    async def generate_speech(
        self,
        text: str,
        voice_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Generate speech from text using Bark"""
        if not self.is_initialized:
            await self.initialize()

        # Set voice if specified
        if voice_name:
            self.set_voice(voice_name)
        elif self.current_voice is None:
            self.set_voice("default")

        # Check cache
        cache_key = self._get_cache_key(text, self.current_voice["name"])
        if use_cache and cache_key in self.audio_cache:
            logger.debug(f"Using cached audio for: {text[:30]}...")
            return self.audio_cache[cache_key]

        try:
            # Generate audio
            with torch.cuda.device(self.device):
                audio_array = generate_audio(
                    text,
                    history_prompt=self.current_voice["voice_id"],
                    text_temp=0.7,
                    waveform_temp=0.7,
                )

            # Cache the generated audio
            if use_cache:
                self.audio_cache[cache_key] = audio_array
                self._cleanup_cache()

            return audio_array

        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            raise

    async def generate_speech_segments(
        self,
        text: str,
        max_segment_length: int = 200,
        voice_name: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Generate speech for long text by splitting into segments"""
        segments = self._split_text(text, max_segment_length)
        audio_segments = []

        for segment in segments:
            audio = await self.generate_speech(segment, voice_name)
            audio_segments.append(audio)

        return audio_segments

    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into segments at sentence boundaries"""
        words = text.split()
        segments = []
        current_segment = []
        current_length = 0

        for word in words:
            word_length = len(word)

            if current_length + word_length + 1 > max_length:
                # Find sentence boundary
                segment = " ".join(current_segment)
                if "." in segment:
                    split_idx = segment.rindex(".")
                    first_part = segment[: split_idx + 1]
                    remainder = segment[split_idx + 1 :] + " " + word

                    segments.append(first_part.strip())
                    current_segment = remainder.strip().split()
                    current_length = len(remainder)
                else:
                    segments.append(segment)
                    current_segment = [word]
                    current_length = word_length
            else:
                current_segment.append(word)
                current_length += word_length + 1

        if current_segment:
            segments.append(" ".join(current_segment))

        return segments

    def save_audio(self, audio_array: np.ndarray, output_path: str) -> None:
        """Save generated audio to file"""
        try:
            import soundfile as sf

            sf.write(output_path, audio_array, self.sample_rate)
            logger.info(f"Audio saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available voice presets"""
        return self.voice_presets

    def get_current_voice(self) -> Dict[str, Any]:
        """Get current voice settings"""
        return self.current_voice or self.voice_presets["default"]


# Global TTS instance
bark_tts = BarkTTS()
