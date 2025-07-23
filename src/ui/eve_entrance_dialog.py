"""
Eve Entrance Dialog
Cinematic entrance with photorealistic 3D avatar and voice greeting
"""
import os
import asyncio
import numpy as np
from tempfile import NamedTemporaryFile
import soundfile as sf  # Assuming soundfile is installed for saving audio
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from loguru import logger
from src.agents.neural_radiance_agent import NeuralRadianceAgent
from src.agents.bark_voice_agent import BarkVoiceAgent
from src.core.config import config

class EveEntranceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: black;")
        self.setFixedSize(1920, 1080)  # Full HD for cinematic feel
        self.layout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.label.setScaledContents(True)
        self.layout.addWidget(self.label, alignment=Qt.AlignCenter)
        self.neural_agent = NeuralRadianceAgent()
        self.voice_agent = BarkVoiceAgent()
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.frame = 0
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.temp_audio = None
        # Initialize agents
        self.loop.run_until_complete(self.neural_agent._initialize())
        self.loop.run_until_complete(self.voice_agent._initialize())
        self.start_entrance()

    def start_entrance(self):
        self.timer.start(16)  # 60fps
        user_name = config.get("user_name", "Creator")
        greeting = f"Welcome to Aisis: The Birth of Celestial Art, {user_name}!"
        task = {"text": greeting, "voice_pack": "cyber-sorceress", "tone": "epic"}
        audio_result = self.loop.run_until_complete(self.voice_agent._process(task))
        audio_data = audio_result["audio"]
        # Save audio to temp file (assuming audio_data is numpy array of samples)
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, samplerate=24000)  # Assuming Bark sample rate
            self.temp_audio = temp_file.name
        self.player.setSource(QUrl.fromLocalFile(self.temp_audio))
        self.player.play()

    def update_frame(self):
        self.frame += 1
        pose = np.array([np.sin(self.frame * 0.05), np.cos(self.frame * 0.05), 5.0])  # Simple rotation
        style = config.get("eve_style", "cyber-sorceress")  # From config or user preference
        image = self.loop.run_until_complete(self.neural_agent.render_avatar(style, pose))
        height, width, _ = image.shape
        qimage = QImage(image, width, height, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimage))
        if self.frame > 600 or self.player.mediaStatus() == QMediaPlayer.EndOfMedia:  # 10s or audio done
            self.timer.stop()
            self.accept()
            if self.temp_audio:
                os.unlink(self.temp_audio)

    def closeEvent(self, event):
        self.loop.run_until_complete(self.neural_agent._cleanup())
        self.loop.run_until_complete(self.voice_agent._cleanup())
        self.loop.close()
        super().closeEvent(event)