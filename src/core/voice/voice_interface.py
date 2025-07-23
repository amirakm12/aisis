import threading
import queue
import pyaudio
import numpy as np
from .faster_whisper_asr import FasterWhisperASR
from .bark_tts import BarkTTS
from loguru import logger

class VoiceInterface:
    def __init__(self, on_command_callback):
        self.asr = FasterWhisperASR(model_size="base")
        self.tts = BarkTTS()
        self.on_command_callback = on_command_callback
        self.audio_queue = queue.Queue()
        self.listening = False
        self.stream = None
        self.thread = None
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.asr.initialize()

    async def initialize(self):
        await self.tts.initialize()

    def start_listening(self):
        if self.listening:
            return
        self.listening = True
        p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
        self.asr.transcribe_stream(self.audio_queue, self.sample_rate, self.chunk_size * 2, self._on_partial, self._on_final)
        logger.info("Voice listening started")

    def _audio_loop(self):
        while self.listening:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)
            self.audio_queue.put(audio)

    def stop_listening(self):
        if not self.listening:
            return
        self.listening = False
        if self.thread:
            self.thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.asr.stop()
        logger.info("Voice listening stopped")

    def _on_partial(self, text):
        logger.debug(f"Partial transcript: {text}")

    def _on_final(self, text):
        if text.strip():
            logger.info(f"Voice command: {text}")
            self.on_command_callback(text)
            self.speak("Command received: " + text)

    def speak(self, text):
        import sounddevice as sd
        audio = self.tts.generate_speech(text)
        sd.play(audio, samplerate=self.tts.sample_rate)
        sd.wait()