"""
AISIS Main Window
Implements the primary GUI interface with voice interaction and image editing
"""

from pathlib import Path
from typing import Optional, Callable
import asyncio
import threading

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QStatusBar, QMessageBox, QListWidget, QListWidgetItem,
    QDialog, QFormLayout, QLineEdit, QComboBox, QDialogButtonBox, QTextEdit
)
from PySide6.QtCore import Qt, QSize, Signal, Slot, QThread, QTime
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QMouseEvent
from loguru import logger

from ..core.voice_manager import voice_manager
from ..core.llm_manager import LLMManager
from src.agents.multi_agent_orchestrator import MultiAgentOrchestrator
from src.agents.llm_meta_agent import LLMMetaAgent
from src.agents.image_restoration import ImageRestorationAgent
from src.agents.style_aesthetic import StyleAestheticAgent
from src.agents.denoising import DenoisingAgent
from src.agents.text_recovery import TextRecoveryAgent
from src.agents.meta_correction import MetaCorrectionAgent
from src.agents.semantic_editing import SemanticEditingAgent
from src.agents.auto_retouch import AutoRetouchAgent
from src.agents.generative import GenerativeAgent
from src.agents.neural_radiance import NeuralRadianceAgent
from src.agents.super_resolution import SuperResolutionAgent
from src.agents.color_correction import ColorCorrectionAgent
from src.agents.tile_stitching import TileStitchingAgent
from src.agents.feedback_loop import FeedbackLoopAgent
from src.agents.perspective_correction import PerspectiveCorrectionAgent
from src.agents.material_recognition import MaterialRecognitionAgent
from src.agents.damage_classifier import DamageClassifierAgent
from src.agents.hyperspectral_recovery import HyperspectralRecoveryAgent
from src.agents.paint_layer_decomposition import PaintLayerDecompositionAgent
from src.agents.self_critique import SelfCritiqueAgent
from src.agents.forensic_analysis import ForensicAnalysisAgent
from src.agents.context_aware_restoration import ContextAwareRestorationAgent
from src.agents.adaptive_enhancement import AdaptiveEnhancementAgent
from src.agents.vision_language import VisionLanguageAgent
from src.agents.style_transfer import StyleTransferAgent
from .context_panel import ContextPanel
from .context_manager import ContextManager
from src.agents.workflow_builder import WorkflowBuilder
from .chat_panel import ChatPanel

class AsyncWorker(QThread):
    """Worker thread for async operations"""
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, coro: Callable, *args, **kwargs):
        super().__init__()
        self.coro = coro
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.coro(*self.args, **self.kwargs))
            loop.close()
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"AsyncWorker error: {e}")
            self.error.emit(str(e))

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QFormLayout(self)
        self.llm_model = QLineEdit()
        self.llm_model.setPlaceholderText("llama-2-7b-chat.gguf")
        layout.addRow("LLM Model:", self.llm_model)
        self.voice_device = QComboBox()
        self.voice_device.addItems(["Default", "CPU", "GPU"])
        layout.addRow("Voice Device:", self.voice_device)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        layout.addRow(self.save_btn)

class DrawingCanvas(QWidget):
    """Widget for freehand drawing/sketch input."""
    sketch_made = Signal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(512, 512)
        self.setStyleSheet("background: #222;")
        self.drawing = False
        self.last_point = None
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(QColor("#222"))
        self.pen = QPen(QColor("#00ff00"), 4)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and self.last_point:
            painter = QPainter(self.image)
            painter.setPen(self.pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.sketch_made.emit(self.image.copy())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def clear(self):
        self.image.fill(QColor("#222"))
        self.update()

class SolutionsDialog(QDialog):
    """Dialog to display multiple solutions and critiques, and get user feedback."""
    def __init__(self, solutions, critiques, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tree-of-Thought Solutions")
        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        text = ""
        for i, sol in enumerate(solutions):
            text += f"Solution {i+1} (Agent: {sol['agent']}):\n{sol['result']}\n\n"
        for i, crit in enumerate(critiques):
            text += f"Critique {i+1}:\n{crit.get('llm_response', crit)}\n\n"
        self.text.setText(text)
        layout.addWidget(self.text)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        layout.addWidget(self.button_box)
        self.feedback = None

    def get_feedback(self):
        # For future: add UI for user to select or comment
        return self.feedback

class MainWindow(QMainWindow):
    """AISIS main application window"""
    
    operation_complete = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AISIS - AI Creative Studio")
        self.setMinimumSize(QSize(1280, 720))
        
        # Context management
        self.context_manager = ContextManager()
        self.context_panel = ContextPanel(self.context_manager)
        
        # Use MultiAgentOrchestrator for advanced reasoning
        self.orchestrator = MultiAgentOrchestrator(meta_agent=LLMMetaAgent())
        # Register all real agents
        try:
            self.orchestrator.register_agent("image_restoration", ImageRestorationAgent())
            self.orchestrator.register_agent("style_aesthetic", StyleAestheticAgent())
            self.orchestrator.register_agent("denoising", DenoisingAgent())
            self.orchestrator.register_agent("text_recovery", TextRecoveryAgent())
            self.orchestrator.register_agent("meta_correction", MetaCorrectionAgent())
            self.orchestrator.register_agent("semantic_editing", SemanticEditingAgent())
            self.orchestrator.register_agent("auto_retouch", AutoRetouchAgent())
            self.orchestrator.register_agent("generative", GenerativeAgent())
            self.orchestrator.register_agent("neural_radiance", NeuralRadianceAgent())
            self.orchestrator.register_agent("super_resolution", SuperResolutionAgent())
            self.orchestrator.register_agent("color_correction", ColorCorrectionAgent())
            self.orchestrator.register_agent("tile_stitching", TileStitchingAgent())
            self.orchestrator.register_agent("feedback_loop", FeedbackLoopAgent())
            self.orchestrator.register_agent("perspective_correction", PerspectiveCorrectionAgent())
            self.orchestrator.register_agent("material_recognition", MaterialRecognitionAgent())
            self.orchestrator.register_agent("damage_classifier", DamageClassifierAgent())
            self.orchestrator.register_agent("hyperspectral_recovery", HyperspectralRecoveryAgent())
            self.orchestrator.register_agent("paint_layer_decomposition", PaintLayerDecompositionAgent())
            self.orchestrator.register_agent("self_critique", SelfCritiqueAgent())
            self.orchestrator.register_agent("forensic_analysis", ForensicAnalysisAgent())
            self.orchestrator.register_agent("context_aware_restoration", ContextAwareRestorationAgent())
            self.orchestrator.register_agent("adaptive_enhancement", AdaptiveEnhancementAgent())
            self.orchestrator.register_agent("vision_language", VisionLanguageAgent())
            self.orchestrator.register_agent("style_transfer", StyleTransferAgent())
        except Exception as e:
            print(f"[ERROR] Failed to register agents: {e}")
        self.current_image: Optional[Path] = None
        self.workers = []
        self.llm_manager = LLMManager()
        self.conversation = []
        self.workflow_builder = WorkflowBuilder(self.orchestrator.agents)
        self.auto_pilot = False
        self.chat_panel = ChatPanel()
        self.chat_panel.workflow_refined.connect(self._run_workflow)
        self.chat_panel.undo_requested.connect(self._on_undo)
        self.chat_panel.branch_requested.connect(self._run_workflow)
        
        self._setup_ui()
        self._initialize_async()
    
        # Warn if running in CPU mode
        import torch
        if not torch.cuda.is_available():
            self.status_bar.showMessage("Warning: CUDA GPU not detected. Running in CPU mode.")
    
    def _setup_ui(self):
        """Setup UI components"""
        # Central widget and main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        # Add context panel as sidebar
        main_layout.addWidget(self.context_panel, 1)
        # Add the rest of the main UI (e.g., editor, chat, etc.)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(QSize(512, 512))
        self.image_label.setStyleSheet("QLabel { background-color: #2a2a2a; }")
        layout.addWidget(self.image_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        button_layout.addWidget(self.open_button)
        
        self.voice_button = QPushButton("Start Voice")
        self.voice_button.clicked.connect(self.toggle_voice)
        button_layout.addWidget(self.voice_button)
        
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        self.settings_button = QPushButton("Settings")
        self.settings_button.setToolTip("Configure voice and LLM options")
        self.settings_button.clicked.connect(self.open_settings)
        button_layout.addWidget(self.settings_button)
        
        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Undo last image edit (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo_edit)
        button_layout.addWidget(self.undo_button)
        
        self.redo_button = QPushButton("Redo")
        self.redo_button.setToolTip("Redo last undone edit (Ctrl+Y)")
        self.redo_button.clicked.connect(self.redo_edit)
        button_layout.addWidget(self.redo_button)
        
        self.clear_chat_button = QPushButton("Clear Chat")
        self.clear_chat_button.setToolTip("Clear the chat history")
        self.clear_chat_button.clicked.connect(self.clear_chat)
        button_layout.addWidget(self.clear_chat_button)
        
        self.export_chat_button = QPushButton("Export Chat")
        self.export_chat_button.setToolTip("Export chat to a text file")
        self.export_chat_button.clicked.connect(self.export_chat)
        button_layout.addWidget(self.export_chat_button)
        
        # Add drawing/sketch canvas (hidden by default)
        self.drawing_canvas = DrawingCanvas()
        self.drawing_canvas.setVisible(False)
        self.drawing_canvas.sketch_made.connect(self._on_sketch_made)
        layout.addWidget(self.drawing_canvas)
        # Add toggle drawing mode button
        self.drawing_button = QPushButton("Draw/Sketch")
        self.drawing_button.setCheckable(True)
        self.drawing_button.setToolTip("Toggle drawing/sketch mode")
        self.drawing_button.clicked.connect(self.toggle_drawing_mode)
        button_layout.addWidget(self.drawing_button)
        
        # Add Tree-of-Thought button
        self.tot_button = QPushButton("Tree-of-Thought Reasoning")
        self.tot_button.setToolTip("Try multiple agents, critique, and pick best")
        self.tot_button.clicked.connect(self.run_tree_of_thought)
        self.tot_button.setEnabled(True)
        button_layout.addWidget(self.tot_button)
        
        # Add natural language command input
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Type a command (e.g., 'Restore and upscale this image')...")
        self.command_input.returnPressed.connect(self._on_command_submit)
        self.command_btn = QPushButton("Run Command")
        self.command_btn.clicked.connect(self._on_command_submit)
        self.auto_pilot_btn = QPushButton("Auto-pilot: Off")
        self.auto_pilot_btn.setCheckable(True)
        self.auto_pilot_btn.clicked.connect(self._toggle_auto_pilot)
        # Add to main layout (top bar)
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.command_input)
        top_bar.addWidget(self.command_btn)
        top_bar.addWidget(self.auto_pilot_btn)
        layout.addLayout(top_bar)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Add chat panel for conversation history
        self.chat_panel = QListWidget()
        self.chat_panel.setMinimumHeight(200)
        self.chat_panel.setStyleSheet("QListWidget { background: #181818; color: #e0e0e0; font-size: 15px; border-radius: 8px; }")
        layout.addWidget(self.chat_panel)
        
        # Add listening indicator
        self.listening_label = QLabel("Not Listening")
        self.listening_label.setAlignment(Qt.AlignCenter)
        self.listening_label.setStyleSheet("QLabel { color: #00ff00; font-weight: bold; font-size: 16px; }")
        self.listening_label.setVisible(False)
        layout.addWidget(self.listening_label)
        
        # Add waveform/audio level indicator
        self.waveform_bar = QProgressBar()
        self.waveform_bar.setRange(0, 100)
        self.waveform_bar.setTextVisible(False)
        self.waveform_bar.setVisible(False)
        layout.addWidget(self.waveform_bar)
        
        # Add chat input box for text commands
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a command or question and press Enter...")
        self.chat_input.returnPressed.connect(self._on_chat_input)
        self.chat_input.setToolTip("Send a text command to the orchestrator")
        layout.addWidget(self.chat_input)
        
        main_layout.addWidget(central_widget)
        self.setCentralWidget(main_widget)
        
        # Add chat panel as a dockable widget or sidebar
        self.chat_dock = QWidget()
        chat_layout = QVBoxLayout()
        self.chat_dock.setLayout(chat_layout)
        chat_layout.addWidget(self.chat_panel)
        self.layout().addWidget(self.chat_dock)
    
    def _initialize_async(self):
        """Initialize async components"""
        worker = AsyncWorker(self._async_init)
        worker.finished.connect(self._on_init_complete)
        worker.error.connect(self._on_init_error)
        worker.start()
        self.workers.append(worker)
    
    async def _async_init(self):
        """Async initialization"""
        await self.orchestrator.initialize()
        await voice_manager.initialize()
    
    @Slot(object)
    def _on_init_complete(self, _):
        """Handle initialization complete"""
        logger.info("AISIS initialization complete")
        self.status_bar.showMessage("Ready")
    
    @Slot(str)
    def _on_init_error(self, error):
        """Handle initialization error"""
        logger.error(f"Initialization error: {error}")
        QMessageBox.critical(self, "Error", f"Initialization failed: {error}")
    
    def open_image(self):
        """Open image file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            self.load_image(Path(file_path))
    
    def load_image(self, path: Path):
        """Load and display image"""
        if not path.exists():
            logger.error(f"Image not found: {path}")
            return
        
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            logger.error(f"Failed to load image: {path}")
            return
        
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled)
        self.current_image = path
        self.save_button.setEnabled(True)
        self.status_bar.showMessage(f"Loaded: {path.name}")
    
    def save_image(self):
        """Save image file dialog"""
        if not self.current_image:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            str(Path.home() / f"edited_{self.current_image.name}"),
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            pixmap = self.image_label.pixmap()
            if pixmap and pixmap.save(file_path):
                self.status_bar.showMessage(f"Saved: {Path(file_path).name}")
            else:
                logger.error(f"Failed to save image: {file_path}")
    
    def toggle_voice(self):
        """Toggle voice interaction"""
        if self.voice_button.text() == "Start Voice":
            self.start_voice()
        else:
            self.stop_voice()
    
    def start_voice(self):
        """Start voice interaction"""
        self.voice_button.setText("Stop Voice")
        self.status_bar.showMessage("Listening...")
        self.listening_label.setText("Listening...")
        self.listening_label.setVisible(True)
        self.waveform_bar.setVisible(True)
        self._live_transcript = ""
        self._voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
        self._voice_thread.start()
    
    def stop_voice(self):
        """Stop voice interaction"""
        self.voice_button.setText("Start Voice")
        self.status_bar.showMessage("Voice stopped")
        self.listening_label.setText("Not Listening")
        self.listening_label.setVisible(False)
        self.waveform_bar.setVisible(False)
        if hasattr(self, '_voice_thread'):
            voice_manager.stop_voice_loop()
    
    def _voice_loop(self):
        self._live_transcript = ""
        def on_command(text):
            self._append_chat(f"You: {text}", user=True)
            self.conversation.append((text, None))
            self._handle_user_intent(text, input_mode="voice")
        # Patch voice_manager to provide live transcription and waveform
        def on_audio_level(level):
            self._update_waveform(level)
        def on_partial_transcript(partial):
            self._update_live_transcript(partial)
        voice_manager.start_voice_loop(on_command=on_command, on_audio_level=on_audio_level, on_partial_transcript=on_partial_transcript)

    def _append_chat(self, message, user=False, thinking=False):
        timestamp = QTime.currentTime().toString("HH:mm:ss")
        prefix = "🧑 " if user else "🤖 "
        icon = "🟢" if user else (
            "💬" if not thinking else "⏳"
        )
        item = QListWidgetItem(
            f"{icon} {prefix}[{timestamp}] {message}"
        )
        if thinking:
            # Add a simple spinner/animation (replace with real animation if desired)
            item.setBackground(QColor("#222244"))
        self.chat_panel.addItem(item)
        self.chat_panel.scrollToBottom()

    def _update_live_transcript(self, partial):
        # Show live transcript as a temporary chat item
        self._live_transcript = partial
        if partial:
            if self.chat_panel.count() and self.chat_panel.item(self.chat_panel.count()-1).text().startswith("🧑 ["):
                self.chat_panel.item(self.chat_panel.count()-1).setText(f"🧑 [Live] {partial}")
            else:
                self._append_chat(f"[Live] {partial}", user=True)

    def _update_waveform(self, level):
        # Update waveform/audio level indicator
        self.waveform_bar.setValue(int(level * 100))

    def _parse_intent(self, text):
        # Very basic intent parsing, can be replaced with NLP/LLM
        text_l = text.lower()
        if any(word in text_l for word in ["edit", "make", "change", "dramatic", "vintage", "brighter"]):
            return "edit_image", {"instruction": text}
        elif any(word in text_l for word in ["again", "more", "repeat", "another"]):
            return "followup", {}
        elif any(word in text_l for word in ["hello", "hi", "how are you", "what can you do"]):
            return "chat", {}
        else:
            return "unknown", {}

    def _get_last_edit_instruction(self):
        # Find the last edit instruction from conversation history
        for user, _ in reversed(self.conversation):
            if user:
                return user
        return None

    def _run_image_edit(self, instruction):
        async def edit_and_update():
            from .. import AISIS
            aisis = AISIS()
            await aisis.initialize()
            result = await aisis.edit_image(self.current_image, instruction)
            if result and result.get("output_image"):
                self._append_chat("AISIS: Edit complete.", user=False)
                image = result["output_image"]
                data = image.tobytes("raw", "RGB")
                qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                scaled = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
                self.save_button.setEnabled(True)
            else:
                self._append_chat("AISIS: Edit failed or not recognized.", user=False)
        self.start_operation(edit_and_update)
    
    def start_operation(self, coro: Callable):
        """Start async operation"""
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate progress
        
        worker = AsyncWorker(coro)
        worker.finished.connect(self._on_operation_complete)
        worker.error.connect(self._on_operation_error)
        worker.start()
        
        self.workers.append(worker)
    
    @Slot(object)
    def _on_operation_complete(self, result):
        """Handle operation complete"""
        self.progress.setVisible(False)
        self.operation_complete.emit()
        
        if isinstance(result, dict) and result.get("output_image"):
            # Convert PIL Image to QPixmap and display
            image = result["output_image"]
            data = image.tobytes("raw", "RGB")
            qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled)
            self.save_button.setEnabled(True)
    
    @Slot(str)
    def _on_operation_error(self, error):
        """Handle operation error"""
        self.progress.setVisible(False)
        logger.error(f"Operation error: {error}")
        QMessageBox.warning(self, "Error", f"Operation failed: {error}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop all workers
        for worker in self.workers:
            worker.quit()
            worker.wait()
        
        # Cleanup
        worker = AsyncWorker(self._async_cleanup)
        worker.finished.connect(event.accept)
        worker.error.connect(lambda _: event.accept())
        worker.start()
        self.workers.append(worker)
    
    async def _async_cleanup(self):
        """Async cleanup"""
        await self.orchestrator.cleanup()
        voice_manager.cleanup()

    def _replace_last_chat(self, message, user=False):
        if self.chat_panel.count():
            timestamp = QTime.currentTime().toString("HH:mm:ss")
            prefix = "🧑 " if user else "🤖 "
            icon = "🟢" if user else "💬"
            self.chat_panel.item(self.chat_panel.count()-1).setText(f"{icon} {prefix}[{timestamp}] {message}")

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            # Save settings (stub)
            pass

    def undo_edit(self):
        # TODO: Implement undo logic
        pass

    def redo_edit(self):
        # TODO: Implement redo logic
        pass

    def clear_chat(self):
        self.chat_panel.clear()

    def export_chat(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Chat", "chat.txt", "Text Files (*.txt)")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                for i in range(self.chat_panel.count()):
                    f.write(self.chat_panel.item(i).text() + "\n")

    # Add keyboard shortcuts in __init__
        self.undo_button.setShortcut("Ctrl+Z")
        self.redo_button.setShortcut("Ctrl+Y")
        self.open_button.setShortcut("Ctrl+O")
        self.save_button.setShortcut("Ctrl+S")
        self.settings_button.setShortcut("Ctrl+,")
        self.clear_chat_button.setShortcut("Ctrl+L")
        self.export_chat_button.setShortcut("Ctrl+E")

    # Add tooltips to all major UI elements
        self.open_button.setToolTip("Open an image (Ctrl+O)")
        self.save_button.setToolTip("Save the current image (Ctrl+S)")
        self.voice_button.setToolTip("Start/stop voice interaction")
        self.image_label.setToolTip("Image display area")
        self.chat_panel.setToolTip("Chat history and conversation")
        self.listening_label.setToolTip("Voice listening status")
        self.waveform_bar.setToolTip("Audio input level")

    # Add thinking animation (stub: can be replaced with spinner/animation)
    def _append_chat(self, message, user=False, thinking=False):
        # ... existing code ...
        if thinking:
            # Add a simple animation or spinner (stub)
            pass

    def toggle_drawing_mode(self):
        is_on = self.drawing_button.isChecked()
        self.drawing_canvas.setVisible(is_on)
        self.image_label.setVisible(not is_on)
        if is_on:
            self.status_bar.showMessage(
                "Drawing mode: Sketch on the canvas below."
            )
        else:
            self.status_bar.showMessage("Ready")

    def _on_sketch_made(self, qimage):
        # Real-time feedback: show preview, send to orchestrator/agent, etc.
        self._append_chat(
            "AISIS: Sketch received. Processing...",
            user=False, thinking=True
        )
        # Send sketch to orchestrator for processing
        task = {
            "sketch": qimage,
            "instruction": "process sketch",
            "input_mode": "sketch"
        }
        self.start_operation(lambda: self._process_orchestrator_task(task))

    def handle_gesture_input(self, gesture_data):
        self._append_chat(
            f"AISIS: Gesture input received. Processing...",
            user=False, thinking=True
        )
        # Send gesture to orchestrator for processing
        task = {
            "gesture": gesture_data,
            "instruction": "process gesture",
            "input_mode": "gesture"
        }
        self.start_operation(lambda: self._process_orchestrator_task(task))

    def _handle_user_intent(self, text, input_mode="text"):
        # Improved intent recognition using LLM or rules+LLM hybrid
        # For now, use simple rules, but can be replaced with LLM
        intent, params = self._parse_intent(text)
        task = {"instruction": text, "intent": intent, "params": params, "input_mode": input_mode}
        if intent in ["edit_image", "restore", "style_transfer", "sketch", "gesture"]:
            if self.current_image or input_mode in ("sketch", "gesture"):
                if self.current_image:
                    task["image"] = self.current_image
                self._append_chat("AISIS: Processing your request...", user=False, thinking=True)
                self.start_operation(lambda: self._process_orchestrator_task(task))
            else:
                self._append_chat("AISIS: Please load an image first.", user=False)
        elif intent == "chat":
            self._append_chat("AISIS: ...", user=False, thinking=True)
            def llm_response():
                response = self.llm_manager.chat(text)
                self._replace_last_chat(f"AISIS: {response}", user=False)
            threading.Thread(target=llm_response, daemon=True).start()
        elif intent == "explain":
            self._append_chat("AISIS: Explaining...", user=False, thinking=True)
            def explain_response():
                response = self.llm_manager.explain(text)
                self._replace_last_chat(f"AISIS: {response}", user=False)
            threading.Thread(target=explain_response, daemon=True).start()
        else:
            self._append_chat("AISIS: Sorry, I didn't understand. Please try a different command.", user=False)

    def _process_orchestrator_task(self, task):
        # Unified async handler for orchestrator tasks
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            self.orchestrator.delegate_task(
                task, list(self.orchestrator.agents.keys())
            )
        )
        loop.close()
        # Show result in chat and update UI as needed
        if result and isinstance(result, dict):
            if "output_image" in result:
                image = result["output_image"]
                data = image.tobytes("raw", "RGB")
                qimage = QImage(
                    data, image.width, image.height, QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qimage)
                scaled = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
                self.save_button.setEnabled(True)
            if "meta_critique" in result:
                self._append_chat(
                    f"Meta-agent critique: {result['meta_critique']}", user=False
                )
            if "agent_reports" in result:
                for agent, report in result["agent_reports"].items():
                    self._append_chat(
                        f"{agent}: {report}", user=False
                    )
        else:
            self._append_chat(
                "AISIS: No result or error from orchestrator.", user=False
            )
        return result

    def _on_chat_input(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_input.clear()
        self._append_chat(f"You: {text}", user=True)
        self.conversation.append((text, None))
        self._handle_user_intent(text, input_mode="text")

    def run_tree_of_thought(self):
        # Example: use current image and last instruction, or stub
        if not self.current_image:
            self._append_chat("AISIS: Please load an image first.", user=False)
            return
        last_instruction = self._get_last_edit_instruction() or "restore this image"
        task = {"instruction": last_instruction}
        agent_names = list(self.orchestrator.agents.keys())[:3]  # Pick up to 3 agents
        if not agent_names:
            self._append_chat("AISIS: No agents registered for ToT.", user=False)
            return
        self._append_chat("AISIS: Running Tree-of-Thought reasoning...", user=False, thinking=True)
        def do_tot():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.orchestrator.tree_of_thought_reasoning(
                    task, agent_names, num_solutions=3, ask_user=True
                )
            )
            loop.close()
            return result
        def on_done(result):
            self._replace_last_chat("AISIS: Tree-of-Thought complete.", user=False)
            dlg = SolutionsDialog(result['all_solutions'], result['critiques'], self)
            dlg.exec()
            # Optionally, use dlg.get_feedback() for further learning
        worker = threading.Thread(target=lambda: on_done(do_tot()), daemon=True)
        worker.start()

    # Example hook: update context and progress from operations
    def update_context(self, event):
        self.context_panel.update_context(event)
    def set_progress(self, value, text=None):
        self.context_panel.set_progress(value, text)
    def set_state(self, state):
        self.context_panel.set_state(state)

    def _on_command_submit(self):
        command = self.command_input.text()
        if not command.strip():
            return
        self.context_panel.log(f"[User Command] {command}")
        workflow = self.workflow_builder.build_workflow(command)
        if not workflow:
            self.context_panel.log("[WorkflowBuilder] No tasks found for command.")
            return
        self.context_panel.log(f"[Workflow] {workflow}")
        self._run_workflow(workflow)

    def _run_workflow(self, workflow):
        # Sequentially execute agent tasks in the workflow
        for task in workflow:
            agent_name = task["agent"]
            params = task.get("params", {})
            agent = self.orchestrator.agents.get(agent_name)
            if not agent:
                self.context_panel.log(f"[Error] Agent '{agent_name}' not found.")
                continue
            try:
                result = agent._process(params) if hasattr(agent, '_process') else agent.run(params)
                self.context_panel.log(f"[Result] {agent_name}: {result}")
            except Exception as e:
                self.context_panel.log(f"[Error] {agent_name}: {e}")

    def _toggle_auto_pilot(self):
        self.auto_pilot = not self.auto_pilot
        self.auto_pilot_btn.setText(f"Auto-pilot: {'On' if self.auto_pilot else 'Off'}")
        self.context_panel.log(f"[Auto-pilot] {'Enabled' if self.auto_pilot else 'Disabled'}")

    def _on_undo(self):
        # Undo last workflow or context change
        self.context_panel.log("[Undo] Last workflow/context change undone.")
        # TODO: Implement actual undo logic
