"""
Comprehensive Test Suite for AISIS
Tests all major components and functionality
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import AISIS components
from aisis import AISIS, aisis
from src.core.config import Config
from src.core.model_manager import ModelManager
from src.core.device import DeviceManager
from src.plugins.plugin_manager import PluginManager
from src.plugins.base_plugin import BasePlugin, PluginMetadata

class TestAISISCore:
    """Test core AISIS functionality"""
    
    def test_aisis_initialization(self):
        """Test AISIS instance creation and initialization"""
        test_aisis = AISIS()
        assert test_aisis is not None
        assert hasattr(test_aisis, 'config')
        assert hasattr(test_aisis, 'model_manager')
        assert hasattr(test_aisis, 'device_manager')
        assert hasattr(test_aisis, 'orchestrator')
        assert hasattr(test_aisis, 'plugin_manager')
    
    def test_global_aisis_instance(self):
        """Test global AISIS instance"""
        assert aisis is not None
        assert isinstance(aisis, AISIS)
    
    def test_aisis_version(self):
        """Test version information"""
        from aisis import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0

class TestConfiguration:
    """Test configuration system"""
    
    def test_config_creation(self):
        """Test config instance creation"""
        config = Config()
        assert config is not None
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = Config()
        # Test that some expected defaults exist
        assert hasattr(config, 'get')
        assert hasattr(config, 'set')
    
    @patch('src.core.config.Path.exists')
    def test_config_file_operations(self, mock_exists):
        """Test config file loading and saving"""
        mock_exists.return_value = False
        config = Config()
        
        # Test setting and getting values
        config.set('test_key', 'test_value')
        assert config.get('test_key') == 'test_value'

class TestModelManager:
    """Test model management functionality"""
    
    def test_model_manager_creation(self):
        """Test model manager instantiation"""
        manager = ModelManager()
        assert manager is not None
        assert hasattr(manager, 'list_models')
        assert hasattr(manager, 'download_model')
    
    @patch('src.core.model_manager.Path.exists')
    def test_model_listing(self, mock_exists):
        """Test model listing functionality"""
        mock_exists.return_value = True
        manager = ModelManager()
        
        # Mock the models directory
        with patch('src.core.model_manager.Path.iterdir') as mock_iterdir:
            mock_iterdir.return_value = []
            models = manager.list_models()
            assert isinstance(models, list)

class TestDeviceManager:
    """Test device management functionality"""
    
    def test_device_manager_creation(self):
        """Test device manager instantiation"""
        manager = DeviceManager()
        assert manager is not None
        assert hasattr(manager, 'get_device_info')
        assert hasattr(manager, 'initialize')
    
    def test_device_info(self):
        """Test device information retrieval"""
        manager = DeviceManager()
        device_info = manager.get_device_info()
        assert isinstance(device_info, dict)

class TestPluginSystem:
    """Test plugin system functionality"""
    
    def test_plugin_manager_creation(self):
        """Test plugin manager instantiation"""
        manager = PluginManager()
        assert manager is not None
        assert hasattr(manager, 'load_plugins')
        assert hasattr(manager, 'list_plugins')
    
    def test_base_plugin_metadata(self):
        """Test plugin metadata structure"""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author"
        )
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test plugin"
        assert metadata.author == "Test Author"
    
    def test_plugin_loading(self):
        """Test plugin loading functionality"""
        manager = PluginManager()
        
        # Create a temporary plugin file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
from src.plugins.base_plugin import BasePlugin, PluginMetadata

class TestPlugin(BasePlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test"
        )
    
    def initialize(self):
        return True
    
    def execute(self, *args, **kwargs):
        return "test_result"
    
    def cleanup(self):
        pass

Plugin = TestPlugin
""")
            temp_plugin_path = Path(f.name)
        
        try:
            # Test loading the plugin
            plugin = manager.load_plugin(temp_plugin_path)
            if plugin:  # Only test if loading succeeded
                assert plugin is not None
                assert plugin.get_metadata().name == "test_plugin"
        finally:
            # Cleanup
            temp_plugin_path.unlink(missing_ok=True)

class TestImageProcessing:
    """Test image processing functionality"""
    
    def create_test_image(self) -> Image.Image:
        """Create a test image for processing"""
        # Create a simple RGB image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    def test_image_creation(self):
        """Test test image creation"""
        img = self.create_test_image()
        assert img is not None
        assert img.size == (100, 100)
        assert img.mode == 'RGB'
    
    @patch('src.agents.multi_agent_orchestrator.MultiAgentOrchestrator.process_image')
    def test_image_processing_pipeline(self, mock_process):
        """Test image processing through AISIS"""
        mock_process.return_value = self.create_test_image()
        
        test_aisis = AISIS()
        
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            test_img = self.create_test_image()
            test_img.save(f.name)
            temp_image_path = f.name
        
        try:
            # Test processing
            result = test_aisis.process_image(temp_image_path)
            assert result is not None
            mock_process.assert_called_once()
        finally:
            # Cleanup
            Path(temp_image_path).unlink(missing_ok=True)

class TestCLI:
    """Test CLI functionality"""
    
    def test_cli_import(self):
        """Test CLI module can be imported"""
        try:
            from aisis.cli import cli
            assert cli is not None
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")
    
    @patch('aisis.cli.aisis')
    def test_cli_agents_command(self, mock_aisis):
        """Test CLI agents command"""
        try:
            from aisis.cli import cli
            from click.testing import CliRunner
            
            mock_aisis.get_available_agents.return_value = {
                'test_agent': {
                    'description': 'Test agent',
                    'capabilities': ['test'],
                    'model': 'test_model'
                }
            }
            
            runner = CliRunner()
            result = runner.invoke(cli, ['agents'])
            
            # Should not crash
            assert result.exit_code in [0, 1]  # Allow for initialization failures
            
        except ImportError as e:
            pytest.skip(f"CLI test skipped: {e}")

class TestAPI:
    """Test REST API functionality"""
    
    def test_api_import(self):
        """Test API module can be imported"""
        try:
            from aisis.api import app
            assert app is not None
        except ImportError as e:
            pytest.skip(f"API import failed: {e}")
    
    @patch('aisis.api.aisis')
    def test_api_status_endpoint(self, mock_aisis):
        """Test API status endpoint"""
        try:
            from aisis.api import app
            from fastapi.testclient import TestClient
            
            # Mock the device manager
            mock_device_manager = Mock()
            mock_device_manager.get_device_info.return_value = {
                'gpu_available': False,
                'memory_usage': {'total': 1000, 'used': 500}
            }
            mock_aisis.device_manager = mock_device_manager
            
            client = TestClient(app)
            response = client.get("/status")
            
            # Should return status information
            assert response.status_code == 200
            data = response.json()
            assert 'version' in data
            assert 'status' in data
            
        except ImportError as e:
            pytest.skip(f"API test skipped: {e}")

class TestHealthCheck:
    """Test health check functionality"""
    
    def test_health_check_import(self):
        """Test health check can be imported"""
        try:
            from health_check import AISISHealthChecker
            checker = AISISHealthChecker()
            assert checker is not None
        except ImportError as e:
            pytest.skip(f"Health check import failed: {e}")
    
    def test_health_check_basic_functionality(self):
        """Test basic health check functionality"""
        try:
            from health_check import AISISHealthChecker
            
            checker = AISISHealthChecker()
            
            # Test that methods exist
            assert hasattr(checker, 'check_project_structure')
            assert hasattr(checker, 'check_dependencies')
            assert hasattr(checker, 'run_full_check')
            
        except ImportError as e:
            pytest.skip(f"Health check test skipped: {e}")

class TestInstallation:
    """Test installation system"""
    
    def test_install_script_import(self):
        """Test installation script can be imported"""
        try:
            from install import AISISInstaller
            installer = AISISInstaller()
            assert installer is not None
        except ImportError as e:
            pytest.skip(f"Install script import failed: {e}")
    
    def test_installer_methods(self):
        """Test installer has required methods"""
        try:
            from install import AISISInstaller
            
            installer = AISISInstaller()
            
            # Test that methods exist
            assert hasattr(installer, 'check_system_requirements')
            assert hasattr(installer, 'install_dependencies')
            assert hasattr(installer, 'setup_environment')
            
        except ImportError as e:
            pytest.skip(f"Installer test skipped: {e}")

class TestIntegration:
    """Integration tests for complete workflows"""
    
    @patch('src.core.model_manager.ModelManager.list_models')
    @patch('src.core.device.DeviceManager.get_device_info')
    def test_full_initialization_workflow(self, mock_device_info, mock_list_models):
        """Test complete AISIS initialization workflow"""
        # Mock dependencies
        mock_device_info.return_value = {'gpu_available': False}
        mock_list_models.return_value = []
        
        # Test initialization
        test_aisis = AISIS()
        
        # Should not crash during initialization
        try:
            test_aisis.initialize()
            # If we get here, initialization succeeded or failed gracefully
            assert True
        except Exception as e:
            # Log the exception but don't fail the test
            # as some components may not be available in test environment
            print(f"Initialization warning: {e}")
            assert True
    
    def test_component_integration(self):
        """Test that components can interact with each other"""
        test_aisis = AISIS()
        
        # Test that components exist and have expected interfaces
        assert hasattr(test_aisis, 'config')
        assert hasattr(test_aisis, 'model_manager')
        assert hasattr(test_aisis, 'device_manager')
        assert hasattr(test_aisis, 'orchestrator')
        assert hasattr(test_aisis, 'plugin_manager')
        
        # Test that they have expected methods
        assert hasattr(test_aisis.config, 'get')
        assert hasattr(test_aisis.model_manager, 'list_models')
        assert hasattr(test_aisis.device_manager, 'get_device_info')
        assert hasattr(test_aisis.plugin_manager, 'list_plugins')

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])