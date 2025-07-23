"""
Bark Voice Agent
Uses Bark TTS for dynamic, multilingual voice generation with tones
"""
import torch
from typing import Dict, Any
from loguru import logger
from .base_agent import BaseAgent
from ..core.voice.bark_tts import BarkTTS  # Assuming this is the class name

class BarkVoiceAgent(BaseAgent):
    def __init__(self):
        super().__init__("BarkVoiceAgent")
        self.tts = None
        self.voice_packs = []

    async def _initialize(self) -> None:
        self.tts = BarkTTS()
        # Load 10 pre-loaded voice packs (3GB, stored locally)
        # For now, placeholder list; in reality, load from models/bark
        self.voice_packs = ["cyber-sorceress", "neon-painter", "gothic-muse"] + ["pack_" + str(i) for i in range(7)]
        logger.info("BarkVoiceAgent initialized with {} voice packs".format(len(self.voice_packs)))

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        text = task.get("text", "")
        voice_pack = task.get("voice_pack", "cyber-sorceress")
        tone = task.get("tone", "epic")
        if voice_pack not in self.voice_packs:
            logger.warning(f"Unknown voice pack {voice_pack}, using default")
            voice_pack = "cyber-sorceress"
        # Assuming BarkTTS has a method generate_audio that returns numpy array or bytes
        audio = self.tts.generate_audio(text, voice=voice_pack, tone=tone)
        return {"status": "success", "audio": audio}

    async def _cleanup(self) -> None:
        if self.tts:
            del self.tts