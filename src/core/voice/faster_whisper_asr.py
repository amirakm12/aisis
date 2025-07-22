from faster_whisper import WhisperModel
import numpy as np
import threading


class FasterWhisperASR:
    """
    Streaming ASR using faster-whisper. Supports real-time chunked transcription
    and partial transcript emission.
    """

    def __init__(self, model_size="small", device="auto"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.running = False
        self.thread = None
        self.partial_callback = None
        self.final_callback = None

    def initialize(self):
        self.model = WhisperModel(self.model_size, device=self.device, compute_type="float16")

    def transcribe_stream(
        self, audio_queue, sample_rate, chunk_size, on_partial=None, on_final=None
    ):
        """
        Start streaming transcription from an audio queue.
        on_partial: callback for partial transcript (str)
        on_final: callback for final transcript (str)
        """
        self.partial_callback = on_partial
        self.final_callback = on_final
        self.running = True
        self.thread = threading.Thread(
            target=self._stream_loop, args=(audio_queue, sample_rate, chunk_size), daemon=True
        )
        self.thread.start()

    def _stream_loop(self, audio_queue, sample_rate, chunk_size):
        buffer = np.array([], dtype=np.float32)
        while self.running:
            try:
                chunk = audio_queue.get(timeout=0.1)
                buffer = np.concatenate([buffer, chunk.flatten()])
                if len(buffer) >= chunk_size:
                    process_chunk = buffer[:chunk_size]
                    buffer = buffer[chunk_size:]
                    segments, info = self.model.transcribe(
                        process_chunk,
                        language="en",
                        beam_size=1,
                        word_timestamps=True,
                        vad_filter=True,
                        vad_parameters={"min_speech_duration_ms": 250},
                    )
                    partial = " ".join([seg.text for seg in segments])
                    if self.partial_callback:
                        self.partial_callback(partial)
                    if self.final_callback:
                        self.final_callback(partial)
            except Exception:
                continue

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()


# TODO: Add more advanced VAD, language selection, and error handling as needed.
