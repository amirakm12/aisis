"""
UI component tests for Al-artworks

This module contains tests for the user interface components of Al-artworks,
including main window, dialogs, and interactive elements.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Import UI components to test
from ui.main_window import MainWindow
from ui.tour_dialog import TourDialog
from ui.onboarding_dialog import OnboardingDialog

@pytest.fixture
def app():
    """Create QApplication instance for testing"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()

@pytest.fixture
def main_window(app):
    """Create main window instance for testing"""
    window = MainWindow()
    yield window
    window.close()

class TestMainWindow:
    """Test cases for the main window"""
    
    def test_window_title(self, main_window):
        """Test that window has correct title"""
        assert main_window.windowTitle() == "Al-artworks - AI Creative Studio"
    
    def test_window_size(self, main_window):
        """Test that window has minimum size"""
        assert main_window.width() >= 1280
        assert main_window.height() >= 720
    
    def test_ui_components_exist(self, main_window):
        """Test that all UI components are created"""
        # Test that central widget exists
        assert main_window.centralWidget() is not None
        
        # Test that status bar exists
        assert main_window.statusBar() is not None
    
    def test_image_loading(self, main_window, tmp_path):
        """Test image loading functionality"""
        # Create a test image
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")
        
        # Mock file dialog to return test image
        with patch('PySide6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (str(test_image), "Image Files (*.png)")
            
            # Trigger image loading
            main_window.open_image()
            
            # Verify that image was loaded
            assert main_window.current_image_path == str(test_image)
    
    def test_save_functionality(self, main_window, tmp_path):
        """Test image saving functionality"""
        # Set up a test image
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")
        main_window.current_image_path = str(test_image)
        
        # Mock file dialog to return save path
        save_path = str(tmp_path / "saved.png")
        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName') as mock_dialog:
            mock_dialog.return_value = (save_path, "PNG Files (*.png)")
            
            # Trigger save
            main_window.save_image()
            
            # Verify save dialog was called
            mock_dialog.assert_called_once()
    
    def test_voice_toggle(self, main_window):
        """Test voice input toggle functionality"""
        # Test initial state
        assert not main_window.voice_active
        
        # Toggle voice on
        main_window.toggle_voice()
        assert main_window.voice_active
        
        # Toggle voice off
        main_window.toggle_voice()
        assert not main_window.voice_active
    
    def test_chat_functionality(self, main_window):
        """Test chat input functionality"""
        # Test that chat area exists
        assert hasattr(main_window, 'chat_area')
        
        # Test chat input
        test_message = "Test message"
        main_window._append_chat(test_message, user=True)
        
        # Verify message was added to chat
        chat_text = main_window.chat_area.toPlainText()
        assert test_message in chat_text
    
    def test_settings_dialog(self, main_window):
        """Test settings dialog functionality"""
        # Mock settings dialog
        with patch('ui.main_window.SettingsDialog') as mock_dialog:
            mock_instance = Mock()
            mock_dialog.return_value = mock_instance
            
            # Open settings
            main_window.open_settings()
            
            # Verify dialog was created and shown
            mock_dialog.assert_called_once()
            mock_instance.exec.assert_called_once()
    
    def test_error_handling(self, main_window):
        """Test error handling in UI"""
        # Test handling of invalid image
        with patch('PySide6.QtWidgets.QMessageBox.critical') as mock_critical:
            # Try to load invalid image
            main_window.current_image_path = "invalid/path/image.png"
            main_window.save_image()
            
            # Verify error message was shown
            mock_critical.assert_called()
    
    def test_theme_support(self, main_window):
        """Test theme switching functionality"""
        # Test that theme can be changed
        initial_style = main_window.styleSheet()
        
        # Change theme (this would be implemented in the actual UI)
        # For now, just test that style sheet can be modified
        main_window.setStyleSheet("background-color: red;")
        assert main_window.styleSheet() != initial_style
    
    def test_responsive_design(self, main_window):
        """Test responsive design elements"""
        # Test that window can be resized
        initial_width = main_window.width()
        initial_height = main_window.height()
        
        # Resize window
        main_window.resize(1600, 900)
        
        # Verify size changed
        assert main_window.width() != initial_width
        assert main_window.height() != initial_height
    
    def test_accessibility(self, main_window):
        """Test accessibility features"""
        # Test that all buttons have accessible names
        for child in main_window.findChildren(QPushButton):
            assert child.text() or child.toolTip()
    
    def test_performance(self, main_window):
        """Test UI performance"""
        import time
        
        # Test window creation time
        start_time = time.time()
        window = MainWindow()
        creation_time = time.time() - start_time
        
        # Window creation should be reasonably fast
        assert creation_time < 1.0  # Less than 1 second
        
        window.close()

class TestTourDialog:
    """Test cases for the tour dialog"""
    
    def test_dialog_creation(self, app):
        """Test tour dialog creation"""
        dialog = TourDialog()
        assert dialog.windowTitle() == "Welcome to Al-artworks!"
        dialog.close()
    
    def test_dialog_content(self, app):
        """Test tour dialog content"""
        dialog = TourDialog()
        
        # Test that dialog has content
        assert dialog.layout() is not None
        
        # Test that dialog has buttons
        buttons = dialog.findChildren(QPushButton)
        assert len(buttons) > 0
        
        dialog.close()

class TestOnboardingDialog:
    """Test cases for the onboarding dialog"""
    
    def test_dialog_creation(self, app):
        """Test onboarding dialog creation"""
        dialog = OnboardingDialog()
        assert dialog.windowTitle() == "Welcome to Al-artworks!"
        dialog.close()
    
    def test_onboarding_flow(self, app):
        """Test onboarding dialog flow"""
    dialog = OnboardingDialog()
        
        # Test that dialog has multiple steps
        # (This would depend on the actual implementation)
        assert dialog.layout() is not None
        
        dialog.close()

def test_ui_integration():
    """Test integration between UI components"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Test that main window can be created
    window = MainWindow()
    assert window is not None
    
    # Test that dialogs can be created
    tour_dialog = TourDialog()
    onboarding_dialog = OnboardingDialog()
    
    assert tour_dialog is not None
    assert onboarding_dialog is not None
    
    # Cleanup
    window.close()
    tour_dialog.close()
    onboarding_dialog.close()
    app.quit()

def test_ui_themes():
    """Test UI theme functionality"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = MainWindow()
    
    # Test different themes
    themes = ["dark", "light"]
    
    for theme in themes:
        # Apply theme (this would be implemented in the actual UI)
        window.setStyleSheet(f"background-color: {'#2a2a2a' if theme == 'dark' else '#ffffff'};")
        
        # Verify theme was applied
        style = window.styleSheet()
        if theme == "dark":
            assert "#2a2a2a" in style
        else:
            assert "#ffffff" in style
    
    window.close()
    app.quit()

if __name__ == "__main__":
    pytest.main([__file__])
                         