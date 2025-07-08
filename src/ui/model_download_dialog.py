"""
Model Download Dialog
Provides a user-friendly interface for downloading and managing AI models
"""

from typing import Dict, Any, List
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QProgressBar, QListWidget, QListWidgetItem, QTextEdit,
    QGroupBox, QFormLayout, QComboBox, QCheckBox, QSpacerItem,
    QSizePolicy, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QWidget
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, Slot
from PySide6.QtGui import QPixmap, QIcon, QFont
from loguru import logger

from ..core.enhanced_model_manager import enhanced_model_manager, ModelStatus

class ModelDownloadWorker(QThread):
    """Worker thread for model downloads"""
    progress_updated = Signal(str, float)  # model_name, progress
    download_completed = Signal(str, bool)  # model_name, success
    status_updated = Signal(str, str)  # model_name, status
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        
    def run(self):
        """Run the download process"""
        try:
            # Add progress callback
            def progress_callback(progress: float):
                self.progress_updated.emit(self.model_name, progress)
            
            enhanced_model_manager.add_progress_callback(self.model_name, progress_callback)
            
            # Start download
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(
                enhanced_model_manager.download_model(self.model_name)
            )
            loop.close()
            
            self.download_completed.emit(self.model_name, success)
            
        except Exception as e:
            logger.error(f"Download worker error: {e}")
            self.download_completed.emit(self.model_name, False)

class ModelInfoWidget(QWidget):
    """Widget to display model information"""
    
    def __init__(self, model_info, parent=None):
        super().__init__(parent)
        self.model_info = model_info
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the model info widget UI"""
        layout = QVBoxLayout(self)
        
        # Model name and status
        header_layout = QHBoxLayout()
        
        name_label = QLabel(self.model_info.name)
        name_font = QFont()
        name_font.setBold(True)
        name_font.setPointSize(12)
        name_label.setFont(name_font)
        header_layout.addWidget(name_label)
        
        # Status badge
        status_label = QLabel(self.model_info.status.value.replace('_', ' ').title())
        status_label.setStyleSheet(self._get_status_style(self.model_info.status))
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setMaximumWidth(100)
        header_layout.addWidget(status_label)
        
        layout.addLayout(header_layout)
        
        # Description
        desc_label = QLabel(self.model_info.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; margin: 5px 0;")
        layout.addWidget(desc_label)
        
        # Details
        details_layout = QFormLayout()
        details_layout.addRow("Type:", QLabel(self.model_info.model_type.replace('_', ' ').title()))
        details_layout.addRow("Size:", QLabel(f"{self.model_info.size_gb:.1f} GB"))
        details_layout.addRow("Capabilities:", QLabel(", ".join(self.model_info.capabilities)))
        
        if self.model_info.status == ModelStatus.DOWNLOADING:
            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(self.model_info.download_progress * 100))
            details_layout.addRow("Progress:", self.progress_bar)
        
        layout.addLayout(details_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        if self.model_info.status == ModelStatus.NOT_DOWNLOADED:
            self.download_btn = QPushButton("Download")
            self.download_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
            button_layout.addWidget(self.download_btn)
        
        elif self.model_info.status == ModelStatus.DOWNLOADED:
            self.load_btn = QPushButton("Load")
            self.load_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
            button_layout.addWidget(self.load_btn)
            
            self.unload_btn = QPushButton("Remove")
            self.unload_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
            button_layout.addWidget(self.unload_btn)
        
        elif self.model_info.status == ModelStatus.LOADED:
            self.unload_btn = QPushButton("Unload")
            self.unload_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
            button_layout.addWidget(self.unload_btn)
        
        layout.addLayout(button_layout)
        
        # Set fixed height for consistent layout
        self.setFixedHeight(150)
        self.setStyleSheet("ModelInfoWidget { border: 1px solid #ddd; border-radius: 5px; margin: 2px; }")
    
    def _get_status_style(self, status: ModelStatus) -> str:
        """Get styling for status label"""
        styles = {
            ModelStatus.NOT_DOWNLOADED: "background-color: #9E9E9E; color: white; padding: 2px 8px; border-radius: 10px;",
            ModelStatus.DOWNLOADING: "background-color: #FF9800; color: white; padding: 2px 8px; border-radius: 10px;",
            ModelStatus.DOWNLOADED: "background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 10px;",
            ModelStatus.LOADING: "background-color: #2196F3; color: white; padding: 2px 8px; border-radius: 10px;",
            ModelStatus.LOADED: "background-color: #009688; color: white; padding: 2px 8px; border-radius: 10px;",
            ModelStatus.ERROR: "background-color: #f44336; color: white; padding: 2px 8px; border-radius: 10px;",
        }
        return styles.get(status, "")

class ModelDownloadDialog(QDialog):
    """Dialog for downloading and managing AI models"""
    
    model_downloaded = Signal(str)  # model_name
    model_loaded = Signal(str)     # model_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Model Manager")
        self.setModal(True)
        self.resize(800, 600)
        
        self.download_workers: Dict[str, ModelDownloadWorker] = {}
        self.model_widgets: Dict[str, ModelInfoWidget] = {}
        
        self._setup_ui()
        self._load_models()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(1000)  # Update every second
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("AI Model Manager")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(16)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)
        
        # Info label
        info_label = QLabel("Download and manage AI models for AISIS agents")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # All models tab
        self.all_models_tab = QWidget()
        self._setup_all_models_tab()
        self.tab_widget.addTab(self.all_models_tab, "All Models")
        
        # Downloaded models tab
        self.downloaded_tab = QWidget()
        self._setup_downloaded_tab()
        self.tab_widget.addTab(self.downloaded_tab, "Downloaded")
        
        # System info tab
        self.system_tab = QWidget()
        self._setup_system_tab()
        self.tab_widget.addTab(self.system_tab, "System Info")
        
        layout.addWidget(self.tab_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_models)
        button_layout.addWidget(self.refresh_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def _setup_all_models_tab(self):
        """Setup the all models tab"""
        layout = QVBoxLayout(self.all_models_tab)
        
        # Filter controls
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Filter by capability:"))
        self.capability_filter = QComboBox()
        self.capability_filter.addItem("All")
        self.capability_filter.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.capability_filter)
        
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # Models list
        self.models_list = QListWidget()
        self.models_list.setSpacing(5)
        layout.addWidget(self.models_list)
    
    def _setup_downloaded_tab(self):
        """Setup the downloaded models tab"""
        layout = QVBoxLayout(self.downloaded_tab)
        
        self.downloaded_list = QListWidget()
        layout.addWidget(self.downloaded_list)
        
        # Bulk actions
        actions_layout = QHBoxLayout()
        
        self.load_all_btn = QPushButton("Load All")
        self.load_all_btn.clicked.connect(self._load_all_models)
        actions_layout.addWidget(self.load_all_btn)
        
        self.unload_all_btn = QPushButton("Unload All")
        self.unload_all_btn.clicked.connect(self._unload_all_models)
        actions_layout.addWidget(self.unload_all_btn)
        
        actions_layout.addStretch()
        
        layout.addLayout(actions_layout)
    
    def _setup_system_tab(self):
        """Setup the system info tab"""
        layout = QVBoxLayout(self.system_tab)
        
        # System info
        self.system_info = QTextEdit()
        self.system_info.setReadOnly(True)
        layout.addWidget(self.system_info)
        
        self._update_system_info()
    
    def _load_models(self):
        """Load model information"""
        try:
            models = enhanced_model_manager.list_models()
            
            # Clear existing
            self.models_list.clear()
            self.model_widgets.clear()
            
            # Get all capabilities for filter
            capabilities = set()
            for model in models:
                capabilities.update(model.capabilities)
            
            # Update capability filter
            self.capability_filter.clear()
            self.capability_filter.addItem("All")
            for cap in sorted(capabilities):
                self.capability_filter.addItem(cap)
            
            # Add model widgets
            for model in models:
                widget = ModelInfoWidget(model)
                self.model_widgets[model.name] = widget
                
                # Connect buttons
                if hasattr(widget, 'download_btn'):
                    widget.download_btn.clicked.connect(
                        lambda checked, name=model.name: self._download_model(name)
                    )
                
                if hasattr(widget, 'load_btn'):
                    widget.load_btn.clicked.connect(
                        lambda checked, name=model.name: self._load_model(name)
                    )
                
                if hasattr(widget, 'unload_btn'):
                    widget.unload_btn.clicked.connect(
                        lambda checked, name=model.name: self._unload_model(name)
                    )
                
                # Add to list
                item = QListWidgetItem()
                item.setSizeHint(widget.sizeHint())
                self.models_list.addItem(item)
                self.models_list.setItemWidget(item, widget)
            
            self._update_downloaded_tab()
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load models: {e}")
    
    def _apply_filter(self, capability: str):
        """Apply capability filter"""
        for i in range(self.models_list.count()):
            item = self.models_list.item(i)
            widget = self.models_list.itemWidget(item)
            
            if capability == "All":
                item.setHidden(False)
            else:
                should_show = capability in widget.model_info.capabilities
                item.setHidden(not should_show)
    
    def _update_downloaded_tab(self):
        """Update the downloaded models tab"""
        self.downloaded_list.clear()
        
        downloaded_models = [
            model for model in enhanced_model_manager.list_models()
            if model.status in [ModelStatus.DOWNLOADED, ModelStatus.LOADED]
        ]
        
        for model in downloaded_models:
            item_text = f"{model.name} ({model.status.value.replace('_', ' ').title()})"
            self.downloaded_list.addItem(item_text)
    
    def _update_system_info(self):
        """Update system information"""
        try:
            import torch
            import platform
            
            info_text = f"""System Information:
            
Platform: {platform.platform()}
Python: {platform.python_version()}
PyTorch: {torch.__version__ if 'torch' in globals() else 'Not installed'}
CUDA Available: {torch.cuda.is_available() if 'torch' in globals() else 'Unknown'}
GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}

Memory Usage:
{self._get_memory_info()}

Models Directory: {enhanced_model_manager.models_dir}
            """
            
            self.system_info.setText(info_text)
            
        except Exception as e:
            self.system_info.setText(f"Error getting system info: {e}")
    
    def _get_memory_info(self) -> str:
        """Get memory usage information"""
        try:
            import psutil
            import torch
            
            cpu_percent = psutil.virtual_memory().percent
            
            memory_info = f"CPU RAM Usage: {cpu_percent:.1f}%"
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_percent = (gpu_memory / gpu_total) * 100
                memory_info += f"\nGPU VRAM Usage: {gpu_percent:.1f}% ({gpu_memory:.1f}GB / {gpu_total:.1f}GB)"
            
            return memory_info
            
        except Exception:
            return "Memory info not available"
    
    def _download_model(self, model_name: str):
        """Start downloading a model"""
        if model_name in self.download_workers:
            QMessageBox.information(self, "Info", "Model is already being downloaded.")
            return
        
        # Confirm download
        model_info = enhanced_model_manager.get_model_info(model_name)
        if model_info:
            reply = QMessageBox.question(
                self, "Confirm Download",
                f"Download {model_name}?\n\nSize: {model_info.size_gb:.1f} GB\nThis may take a while.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                worker = ModelDownloadWorker(model_name)
                worker.progress_updated.connect(self._on_download_progress)
                worker.download_completed.connect(self._on_download_completed)
                
                self.download_workers[model_name] = worker
                worker.start()
    
    def _load_model(self, model_name: str):
        """Load a model into memory"""
        try:
            # This would be implemented with async loading
            QMessageBox.information(self, "Info", f"Loading {model_name}...")
            self.model_loaded.emit(model_name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
    
    def _unload_model(self, model_name: str):
        """Unload a model from memory"""
        try:
            success = enhanced_model_manager.unload_model(model_name)
            if success:
                QMessageBox.information(self, "Success", f"Unloaded {model_name}")
            else:
                QMessageBox.warning(self, "Warning", f"Model {model_name} was not loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to unload model: {e}")
    
    def _load_all_models(self):
        """Load all downloaded models"""
        QMessageBox.information(self, "Info", "Loading all downloaded models...")
        # Implementation would load all downloaded models
    
    def _unload_all_models(self):
        """Unload all loaded models"""
        QMessageBox.information(self, "Info", "Unloading all models...")
        # Implementation would unload all models
    
    def _refresh_models(self):
        """Refresh the model list"""
        self._load_models()
        self._update_system_info()
    
    def _update_display(self):
        """Update the display with current model status"""
        for model_name, widget in self.model_widgets.items():
            model_info = enhanced_model_manager.get_model_info(model_name)
            if model_info and hasattr(widget, 'progress_bar'):
                widget.progress_bar.setValue(int(model_info.download_progress * 100))
    
    @Slot(str, float)
    def _on_download_progress(self, model_name: str, progress: float):
        """Handle download progress updates"""
        if model_name in self.model_widgets:
            widget = self.model_widgets[model_name]
            if hasattr(widget, 'progress_bar'):
                widget.progress_bar.setValue(int(progress * 100))
    
    @Slot(str, bool)
    def _on_download_completed(self, model_name: str, success: bool):
        """Handle download completion"""
        if model_name in self.download_workers:
            del self.download_workers[model_name]
        
        if success:
            QMessageBox.information(self, "Success", f"Successfully downloaded {model_name}")
            self.model_downloaded.emit(model_name)
            self._refresh_models()
        else:
            QMessageBox.critical(self, "Error", f"Failed to download {model_name}")
    
    def closeEvent(self, event):
        """Handle dialog close"""
        # Stop update timer
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        
        # Clean up any running downloads
        for worker in self.download_workers.values():
            if worker.isRunning():
                worker.quit()
                worker.wait(1000)  # Wait up to 1 second
        
        event.accept()