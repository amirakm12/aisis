"""
Whisper ASR (Automatic Speech Recognition) Module
Handles real-time speech-to-text conversion using OpenAI Whisper
"""

import asyncio
import queue
import threading
import time
from typing import Callable, Optional, Dict, Any

import numpy as np
import soundfile as sf
import torch
import whisper
from loguru import logger

from ..config import config
from ..gpu_utils import gpu_manager, model_loader


class WhisperASR:
    """Real-time Whisper ASR with GPU acceleration"""

    def __init__(self, model_size: str = None, language: str = "en"):
        self.model_size = model_size or config.voice.whisper_model
        self.language = language
        self.sample_rate = config.voice.sample_rate
        self.chunk_duration = config.voice.chunk_size

        self.model = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.result_callback: Optional[Callable[[str], None]] = None

        # Audio processing parameters
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.overlap_size = int(self.sample_rate * 1.0)  # 1 second overlap

        logger.info(f"Initializing Whisper ASR with model: {self.model_size}")

    async def initialize(self) -> None:
        """Initialize the Whisper model"""
        try:
            logger.info("Loading Whisper model...")
            self.model = whisper.load_model(self.model_size, device=gpu_manager.device)

            # Optimize model for inference
            self.model = gpu_manager.optimize_model_for_inference(self.model)

            logger.info(f"Whisper model loaded successfully on {gpu_manager.device}")

        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise

    def set_result_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for transcription results"""
        self.result_callback = callback

    async def transcribe_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio data using Whisper"""
        if self.model is None:
            await self.initialize()

        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Pad or trim audio to 30 seconds (Whisper's expected input length)
            target_length = self.sample_rate * 30
            if len(audio_data) > target_length:
                audio_data = audio_data[:target_length]
            else:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))

            # Run inference
            with torch.no_grad():
                result = self.model.transcribe(
                    audio_data,
                    language=self.language,
                    fp16=gpu_manager.device.type == "cuda",
                    verbose=False,
                )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"text": "", "segments": []}

    async def transcribe_file(self, file_path: str) -> str:
        """Transcribe audio from file"""
        try:
            # Load audio file
            audio_data, sample_rate = sf.read(file_path)

            # Resample if necessary
            if sample_rate != self.sample_rate:
                import librosa

                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=self.sample_rate
                )

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            result = await self.transcribe_audio(audio_data)
            return result.get("text", "")

        except Exception as e:
            logger.error(f"Failed to transcribe file {file_path}: {e}")
            return ""

    def start_real_time_transcription(self) -> None:
        """Start real-time audio transcription"""
        if self.is_listening:
            logger.warning("Already listening for audio")
            return

        self.is_listening = True

        # Start audio processing thread
        audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        audio_thread.start()

        logger.info("Started real-time transcription")

    def stop_real_time_transcription(self) -> None:
        """Stop real-time audio transcription"""
        self.is_listening = False
        logger.info("Stopped real-time transcription")

    def _audio_processing_loop(self) -> None:
        """Main audio processing loop for real-time transcription"""
        audio_buffer = np.array([], dtype=np.float32)

        while self.is_listening:
            try:
                # Get audio chunk from queue (with timeout)
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Add to buffer
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])

                # Process when we have enough audio
                if len(audio_buffer) >= self.chunk_size:
                    # Extract chunk for processing
                    process_chunk = audio_buffer[: self.chunk_size]

                    # Keep overlap for next iteration
                    audio_buffer = audio_buffer[self.chunk_size - self.overlap_size :]

                    # Transcribe chunk
                    asyncio.run(self._process_audio_chunk(process_chunk))

            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")

    async def _process_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """Process a single audio chunk"""
        try:
            result = await self.transcribe_audio(audio_chunk)
            text = result.get("text", "").strip()

            if text and self.result_callback:
                self.result_callback(text)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    def add_audio_data(self, audio_data: np.ndarray) -> None:
        """Add audio data to processing queue"""
        if self.is_listening:
            self.audio_queue.put(audio_data)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_size": self.model_size,
            "language": self.language,
            "device": str(gpu_manager.device),
            "sample_rate": self.sample_rate,
            "chunk_duration": self.chunk_duration,
        }


class VoiceActivityDetector:
    """Simple voice activity detection to improve transcription efficiency"""

    def __init__(self, threshold: float = 0.01, min_duration: float = 0.5):
        self.threshold = threshold
        self.min_duration = min_duration
        self.sample_rate = config.voice.sample_rate
        self.min_samples = int(self.min_duration * self.sample_rate)

    def is_speech(self, audio_data: np.ndarray) -> bool:
        """Detect if audio contains speech"""
        # Simple energy-based VAD
        energy = np.mean(audio_data**2)

        # Check if energy is above threshold and duration is sufficient
        return energy > self.threshold and len(audio_data) >= self.min_samples

    def trim_silence(self, audio_data: np.ndarray) -> np.ndarray:
        """Remove silence from beginning and end of audio"""
        # Find first and last non-silent samples
        energy = audio_data**2
        above_threshold = energy > self.threshold

        if not np.any(above_threshold):
            return audio_data  # No speech detected, return original

        first_speech = np.argmax(above_threshold)
        last_speech = len(above_threshold) - 1 - np.argmax(above_threshold[::-1])

        return audio_data[first_speech : last_speech + 1]


# Global ASR instance
whisper_asr = WhisperASR()
vad = VoiceActivityDetector()
