# AISIS Voice Modules

This folder contains all voice-related modules for AISIS, including:
- Text-to-Speech (TTS) with Bark
- Automatic Speech Recognition (ASR) with Whisper and FasterWhisper

## Architecture
- TTS modules (e.g., `bark_tts.py`) synthesize speech from text.
- ASR modules (e.g., `whisper_asr.py`, `faster_whisper_asr.py`) transcribe audio to text.
- The `voice_manager.py` orchestrates TTS and ASR for real-time voice interaction.

## Adding a New Voice Model
1. Add your model as a new module in this folder.
2. Expose a clear, async API for initialization and inference.
3. Register your model in `voice_manager.py` if needed.

## Guidelines
- Use type hints and docstrings for all public methods.
- Keep models modular and easy to swap/upgrade.
- Document any model-specific requirements or dependencies.

---
See each module for implementation details and extension notes. 