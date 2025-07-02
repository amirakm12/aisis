"""
AISIS Application Launcher
Complete application that integrates modern UI, local models, and enhanced agents
into a cohesive, professional system.
"""

import sys
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Core imports
from .core.config import config
from .core.gpu_utils import gpu_manager
from .core.advanced_local_models import local_model_manager

# Agent imports
from .agents import AGENT_REGISTRY, register_agent
from .agents.enhanced_image_restoration import EnhancedImageRestorationAgent
from .agents.multi_agent_orchestrator import MultiAgentOrchestrator

# UI imports
from .ui.modern_interface import (
    create_modern_app, ModernMainWindow, Theme, ModernThemeManager
)

# Integration imports
from .core.integration import INTEGRATION_REGISTRY
from .core.device import get_current_device_adapter


class AISISApplication:
    """
    Main AISIS application class that orchestrates all components.
    Provides a unified interface for the entire system.
    """
    
    def __init__(self):
        self.app = None
        self.main_window = None
        self.orchestrator = None
        self.device_adapter = None
        self.theme_manager = None
        self.initialized = False
        
        # Application state
        self.current_project = None
        self.active_agents = {}
        self.processing_queue = []
        
    async def initialize(self) -> None:
        """Initialize the complete AISIS application"""
        try:
            print("🚀 Initializing AISIS Application...")
            
            # Initialize core systems
            await self._initialize_core_systems()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Initialize UI
            await self._initialize_ui()
            
            # Initialize integrations
            await self._initialize_integrations()
            
            self.initialized = True
            print("✅ AISIS Application initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize AISIS Application: {e}")
            raise
    
    async def _initialize_core_systems(self) -> None:
        """Initialize core systems"""
        print("📦 Initializing core systems...")
        
        # Initialize GPU manager
        gpu_manager.initialize()
        print(f"🎮 GPU: {gpu_manager.device}")
        
        # Initialize device adapter
        self.device_adapter = get_current_device_adapter()
        print(f"📱 Device: {self.device_adapter.name}")
        
        # Initialize theme manager
        self.theme_manager = ModernThemeManager()
        
        # Load configuration
        self._load_configuration()
    
    async def _initialize_agents(self) -> None:
        """Initialize AI agents"""
        print("🤖 Initializing AI agents...")
        
        # Register enhanced agents
        enhanced_restoration = EnhancedImageRestorationAgent()
        register_agent("enhanced_restoration", enhanced_restoration)
        
        # Initialize orchestrator
        self.orchestrator = MultiAgentOrchestrator()
        
        # Register agents with orchestrator
        for name, agent in AGENT_REGISTRY.items():
            self.orchestrator.register_agent(name, agent)
            print(f"  ✅ Registered agent: {name}")
        
        # Initialize all agents
        for agent in AGENT_REGISTRY.values():
            try:
                await agent.initialize()
                self.active_agents[agent.name] = agent
            except Exception as e:
                print(f"  ⚠️ Failed to initialize {agent.name}: {e}")
    
    async def _initialize_ui(self) -> None:
        """Initialize the modern UI"""
        print("🎨 Initializing modern UI...")
        
        # Create QApplication
        self.app = create_modern_app()
        
        # Create main window
        self.main_window = ModernMainWindow()
        
        # Apply theme
        self.theme_manager.apply_theme(self.app, Theme.DARK)
        
        # Connect UI signals
        self._connect_ui_signals()
        
        print("  ✅ Modern UI initialized")
    
    async def _initialize_integrations(self) -> None:
        """Initialize external integrations"""
        print("🔗 Initializing integrations...")
        
        # Initialize available integrations
        for name, integration in INTEGRATION_REGISTRY.items():
            print(f"  📡 Integration available: {name}")
        
        print("  ✅ Integrations initialized")
    
    def _load_configuration(self) -> None:
        """Load application configuration"""
        config_path = Path("config.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                print("  📋 Configuration loaded")
            except Exception as e:
                print(f"  ⚠️ Failed to load config: {e}")
                self.config = self._get_default_config()
        else:
            self.config = self._get_default_config()
            self._save_configuration()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "theme": "dark",
            "gpu_acceleration": True,
            "auto_save": True,
            "max_processing_threads": 4,
            "default_quality": "quality",
            "integrations": {
                "google_drive": False,
                "dropbox": False
            }
        }
    
    def _save_configuration(self) -> None:
        """Save application configuration"""
        try:
            with open("config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save config: {e}")
    
    def _connect_ui_signals(self) -> None:
        """Connect UI signals to application logic"""
        if not self.main_window:
            return
        
        # Connect sidebar navigation
        for i, btn in enumerate(self.main_window.sidebar.nav_buttons):
            btn.clicked.connect(
                lambda checked, index=i: self._handle_navigation(index)
            )
        
        # Connect theme toggle
        self.main_window.sidebar.theme_btn.clicked.connect(self._toggle_theme)
    
    def _handle_navigation(self, index: int) -> None:
        """Handle navigation between different pages"""
        page_names = ["Dashboard", "Image Editor", "AI Agents", "Settings", "Analytics"]
        print(f"📱 Navigating to: {page_names[index]}")
        
        # Update content based on navigation
        if index == 1:  # Image Editor
            self._setup_image_editor()
        elif index == 2:  # AI Agents
            self._setup_agents_page()
        elif index == 3:  # Settings
            self._setup_settings_page()
    
    def _setup_image_editor(self) -> None:
        """Setup the image editor page"""
        print("🖼️ Setting up image editor...")
        # Add image editor specific functionality here
    
    def _setup_agents_page(self) -> None:
        """Setup the AI agents page"""
        print("🤖 Setting up AI agents page...")
        # Add agent management functionality here
    
    def _setup_settings_page(self) -> None:
        """Setup the settings page"""
        print("⚙️ Setting up settings page...")
        # Add settings management functionality here
    
    def _toggle_theme(self) -> None:
        """Toggle between light and dark themes"""
        current_theme = self.theme_manager.current_theme
        new_theme = Theme.LIGHT if current_theme == Theme.DARK else Theme.DARK
        
        self.theme_manager.apply_theme(self.app, new_theme)
        
        # Update config
        self.config["theme"] = new_theme.value
        self._save_configuration()
        
        # Update UI
        btn_text = "🌙 Dark Mode" if new_theme == Theme.LIGHT else "☀️ Light Mode"
        self.main_window.sidebar.theme_btn.setText(btn_text)
    
    async def process_image_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process an image task using the orchestrator"""
        if not self.initialized:
            raise RuntimeError("Application not initialized")
        
        try:
            # Find suitable agents for the task
            task_type = task.get('type', 'image_restoration')
            suitable_agents = [
                name for name, agent in self.active_agents.items()
                if task_type in agent.capabilities.get('tasks', [])
            ]
            
            if not suitable_agents:
                raise ValueError(f"No suitable agents found for task: {task_type}")
            
            # Use orchestrator to process task
            result = await self.orchestrator.delegate_task(task, suitable_agents)
            
            return result
            
        except Exception as e:
            print(f"❌ Task processing failed: {e}")
            raise
    
    def show(self) -> None:
        """Show the main application window"""
        if self.main_window:
            self.main_window.show()
            print("🖥️ Application window displayed")
    
    def run(self) -> int:
        """Run the application"""
        if not self.initialized:
            print("❌ Application not initialized")
            return 1
        
        try:
            print("🚀 Starting AISIS Application...")
            self.show()
            
            # Run the application
            return self.app.exec()
            
        except Exception as e:
            print(f"❌ Application failed to run: {e}")
            return 1
    
    async def cleanup(self) -> None:
        """Cleanup application resources"""
        print("🧹 Cleaning up application...")
        
        # Cleanup agents
        for agent in self.active_agents.values():
            try:
                await agent.cleanup()
            except Exception as e:
                print(f"⚠️ Failed to cleanup {agent.name}: {e}")
        
        # Cleanup orchestrator
        if self.orchestrator:
            # Add orchestrator cleanup if needed
            pass
        
        # Save configuration
        self._save_configuration()
        
        print("✅ Cleanup completed")


def create_aisis_app() -> AISISApplication:
    """Create and configure AISIS application"""
    return AISISApplication()


async def run_aisis_app() -> int:
    """Run the complete AISIS application"""
    app = create_aisis_app()
    
    try:
        # Initialize application
        await app.initialize()
        
        # Run application
        return app.run()
        
    except Exception as e:
        print(f"❌ Application failed: {e}")
        return 1
    
    finally:
        # Cleanup
        await app.cleanup()


def main():
    """Main entry point for the AISIS application"""
    print("🎯 AISIS - Advanced AI Image System")
    print("=" * 50)
    
    # Run the application
    return asyncio.run(run_aisis_app())


if __name__ == "__main__":
    sys.exit(main()) 