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


class VoiceManager:
    """Manages voice input/output and streaming ASR for AISIS."""

    def __init__(self) -> None:
        self.asr: Optional[FasterWhisperASR] = None
        self.tts = BarkTTS()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.initialized = False
        self.last_command: Optional[str] = None
        self.is_listening = False

    async def initialize(self) -> None:
        """Initialize voice models asynchronously."""
        if not self.initialized:
            self.asr = FasterWhisperASR(
                model_size="small",
                device="cuda" if torch.cuda.is_available() else "cpu",
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

    async def synthesize(
        self, text: str, output_path: Path = None, voice_name: str = None
    ) -> Path:
        """Synthesize text to speech using Bark TTS"""
        if not self.initialized:
            await self.initialize()
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string.")
        audio_array = await self.tts.generate_speech(
            text, voice_name=voice_name
        )
        if output_path is None:
            output_path = self.tts.cache_dir / f"response_{hash(text)}.wav"
        self.tts.save_audio(audio_array, str(output_path))
        return output_path

    def start_voice_loop(
        self,
        on_command: Optional[Callable[[str], None]] = None,
        on_audio_level: Optional[Callable[[float], None]] = None,
        on_partial_transcript: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Start a real-time voice command loop using streaming ASR.
        Calls on_command(text) for each recognized command.
        Optionally calls on_audio_level(level) and on_partial_transcript(partial)
        for UI feedback.
        """
        import queue
        import threading
        import sounddevice as sd
        import numpy as np
        import time

        self._check_dependencies()

        if not self.initialized:
            raise RuntimeError(
                "Voice system not initialized. Call initialize() first."
            )

        self.is_listening = True
        audio_queue = queue.Queue()
        sample_rate = 16000  # TODO: Make configurable
        chunk_size = int(sample_rate * 2.0)  # 2 seconds per chunk

        def audio_callback(indata, frames, time_info, status):
            audio_queue.put(indata.copy())
            if on_audio_level:
                level = float(np.linalg.norm(indata) / (len(indata) or 1))
                level = min(level, 1.0)
                on_audio_level(level)

        def on_partial(partial: str):
            if on_partial_transcript:
                on_partial_transcript(partial)

        def on_final(final: str):
            if final and on_command:
                on_command(final)

        def listen_loop():
            print("[Voice] Listening for commands. Press Ctrl+C to stop.")
            self.asr.transcribe_stream(
                audio_queue,
                sample_rate,
                chunk_size,
                on_partial=on_partial,
                on_final=on_final,
            )
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=chunk_size,
            ):
                while self.is_listening:
                    time.sleep(0.1)

        thread = threading.Thread(target=listen_loop, daemon=True)
        thread.start()

    def stop_voice_loop(self) -> None:
        """Stop the real-time voice command loop."""
        self.is_listening = False
        if self.asr:
            self.asr.stop()
        print("[Voice] Voice loop stopped.")

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
