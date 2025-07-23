"""
Voice Manager for AISIS
Handles voice recognition (Whisper) and synthesis (Bark)
"""

import asyncio
from pathlib import Path
import torch
from .config import config
from .voice.bark_tts import BarkTTS
from .voice.faster_whisper_asr import FasterWhisperASR
from typing import Callable, Optional
import threading
import time
import numpy as np
import sounddevice as sd
import queue
from .faster_whisper_asr import FasterWhisperASR
from .bark_tts import BarkTTS
from loguru import logger
import torch  # for cuda check

class VoiceManager:
    """Manages voice input/output and streaming ASR for AISIS."""
    def __init__(self):
        self.asr = FasterWhisperASR(model_size="small", device="cuda" if torch.cuda.is_available() else "cpu")
        self.tts = BarkTTS()
        self.running = False
        self.wake_word = "hey aisis"
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.chunk_size = int(self.sample_rate * 1.0)  # 1 second chunks
        self.on_command = None
        self.on_audio_level = None
        self.on_partial_transcript = None

    async def initialize(self):
        """Initialize voice models asynchronously."""
        if not self.initialized:
            self.asr = FasterWhisperASR(
                model_size="small",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.asr.initialize()
            await self.tts.initialize()
            self.initialized = True

    async def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file to text using Whisper ASR"""
        if not self.initialized:
            await self.initialize()
        if not audio_path or not isinstance(audio_path, (str, Path)):
            raise ValueError("audio_path must be a valid file path.")
        return await self.asr.transcribe_file(str(audio_path))

    async def synthesize(self, text: str, output_path: Path = None, voice_name: str = None) -> Path:
        """Synthesize text to speech using Bark TTS"""
        if not self.initialized:
            await self.initialize()
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string.")
        audio_array = await self.tts.generate_speech(text, voice_name=voice_name)
        if output_path is None:
            output_path = self.tts.cache_dir / f"response_{hash(text)}.wav"
        self.tts.save_audio(audio_array, str(output_path))
        return output_path

    def start_voice_loop(self, on_command, on_audio_level=None, on_partial_transcript=None):
        """
        Start a real-time voice command loop using streaming ASR.
        Calls on_command(text) for each recognized command.
        Optionally calls on_audio_level(level) and on_partial_transcript(partial)
        for UI feedback.
        """
        self.on_command = on_command
        self.on_audio_level = on_audio_level
        self.on_partial_transcript = on_partial_transcript
        self.running = True
        # Start audio input thread (stub: use pyaudio or similar)
        threading.Thread(target=self._audio_input_loop, daemon=True).start()
        # Start processing thread
        threading.Thread(target=self._process_loop, daemon=True).start()

    def _audio_input_loop(self):
        def audio_callback(indata, frames, time_info, status):
            self.audio_queue.put(indata.copy().flatten().astype(np.float32))
            if self.on_audio_level:
                level = np.max(np.abs(indata)) / 32768.0  # Normalize
                self.on_audio_level(level)
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16', blocksize=self.chunk_size, callback=audio_callback):
            while self.running:
                time.sleep(0.1)

    def _process_loop(self):
        buffer = np.array([])
        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                buffer = np.concatenate([buffer, chunk])
                if len(buffer) >= self.chunk_size * 3:  # Process larger buffers for context
                    segments, info = self.asr.model.transcribe(buffer, beam_size=5, language="en")
                    transcript = " ".join([seg.text.lower() for seg in segments])
                    if self.on_partial_transcript:
                        self.on_partial_transcript(transcript)
                    if self.wake_word in transcript:
                        command = transcript.split(self.wake_word, 1)[1].strip()
                        if command and self.on_command:
                            self.on_command(command)
                            self.speak("Command received: " + command)
                    buffer = buffer[-self.chunk_size:]  # Keep some history
                # Audio level stub
                level = np.max(np.abs(chunk))
                if self.on_audio_level:
                    self.on_audio_level(level)
            except Exception as e:
                logger.error(f"Voice processing error: {e}")

    def stop_voice_loop(self) -> None:
        """Stop the real-time voice command loop."""
        self.running = False
        if self.asr:
            self.asr.stop()
        print("[Voice] Voice loop stopped.")

    def speak(self, text):
        audio = self.tts.generate_speech(text)
        sd.play(audio, samplerate=self.tts.sample_rate)
        sd.wait()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.initialized:
            self.asr = None
            self.tts = None
            torch.cuda.empty_cache()
            self.initialized = False
            print("[Voice] Voice system cleanup complete")

    @staticmethod
    def _check_dependencies() -> None:
        try:
            import numpy
            import torch
            import sounddevice
        except ImportError as e:
            raise ImportError(
                f"[VoiceManager] Missing dependency: {e}. Please install all requirements."
            )

voice_manager = VoiceManager()
