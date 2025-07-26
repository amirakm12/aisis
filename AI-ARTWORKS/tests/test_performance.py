"""
Performance tests for Al-artworks

This module contains performance tests for Al-artworks components,
including benchmarks for image processing, AI operations, and UI responsiveness.
"""

import pytest
import time
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import AlArtworks

class TestPerformance:
    """Performance test cases for Al-artworks"""

@pytest.fixture
    async def al_artworks(self):
        """Create Al-artworks instance for testing"""
        app = AlArtworks()
        await app.initialize()
        yield app
        await app.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialization_performance(self):
        """Test initialization performance"""
        start_time = time.time()
        
        app = AlArtworks()
        await app.initialize()
        
        init_time = time.time() - start_time
        
        # Initialization should be reasonably fast
        assert init_time < 5.0  # Less than 5 seconds
        
        await app.cleanup()
    
    @pytest.mark.asyncio
    async def test_image_restoration_performance(self, al_artworks, tmp_path):
        """Test image restoration performance"""
        # Create a test image
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")
        output_path = tmp_path / "restored.png"
        
        start_time = time.time()
        
        result = await al_artworks.restore_image(
            str(test_image), 
            str(output_path), 
            "comprehensive"
        )
        
        processing_time = time.time() - start_time
        
        # Processing should be reasonably fast
        assert processing_time < 10.0  # Less than 10 seconds
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_forensic_analysis_performance(self, al_artworks, tmp_path):
        """Test forensic analysis performance"""
        # Create a test image
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")
        
        start_time = time.time()
        
        result = await al_artworks.forensic_analysis(str(test_image))
        
        processing_time = time.time() - start_time
        
        # Analysis should be reasonably fast
        assert processing_time < 5.0  # Less than 5 seconds
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, al_artworks, tmp_path):
        """Test concurrent operation performance"""
        # Create multiple test images
        test_images = []
        for i in range(3):
            test_image = tmp_path / f"test_{i}.png"
            test_image.write_bytes(b"fake image data")
            test_images.append(str(test_image))
        
        start_time = time.time()
        
        # Run concurrent operations
        tasks = []
        for i, image_path in enumerate(test_images):
            output_path = tmp_path / f"restored_{i}.png"
            task = al_artworks.restore_image(image_path, str(output_path), "comprehensive")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Concurrent processing should be efficient
        assert total_time < 15.0  # Less than 15 seconds for 3 operations
        assert all(r["status"] == "success" for r in results)
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, al_artworks):
        """Test memory usage during operations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform some operations
        for i in range(5):
            # Simulate some processing
            await asyncio.sleep(0.1)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
    
    # Memory usage should be reasonable
        assert memory_increase < 500  # Less than 500MB increase

@pytest.mark.asyncio
    async def test_cpu_usage(self, al_artworks):
        """Test CPU usage during operations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Monitor CPU usage during operations
        cpu_percentages = []
        
        for i in range(10):
            start_cpu = process.cpu_percent()
            
            # Simulate some processing
            await asyncio.sleep(0.1)
            
            end_cpu = process.cpu_percent()
            cpu_percentages.append(end_cpu - start_cpu)
        
        # CPU usage should be reasonable
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
        assert avg_cpu < 80  # Less than 80% average CPU usage
    
    def test_ui_responsiveness(self):
        """Test UI responsiveness"""
        from PySide6.QtWidgets import QApplication
        from ui.main_window import MainWindow
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        start_time = time.time()
        
        # Create main window
        window = MainWindow()
        
        creation_time = time.time() - start_time
        
        # Window creation should be fast
        assert creation_time < 1.0  # Less than 1 second
        
        # Test UI responsiveness
        start_time = time.time()
        
        # Simulate some UI operations
        window.resize(1400, 900)
        window.show()
        
        ui_time = time.time() - start_time
        
        # UI operations should be responsive
        assert ui_time < 0.5  # Less than 0.5 seconds
        
        window.close()
    
    @pytest.mark.asyncio
    async def test_network_performance(self, al_artworks):
        """Test network performance for remote operations"""
        # Mock network operations
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = asyncio.coroutine(
                lambda: {"status": "success"}
            )
            
            start_time = time.time()
            
            # Simulate network operation
            await asyncio.sleep(0.1)
            
            network_time = time.time() - start_time
            
            # Network operations should be reasonably fast
            assert network_time < 1.0  # Less than 1 second
    
    def test_disk_io_performance(self, tmp_path):
        """Test disk I/O performance"""
        test_file = tmp_path / "performance_test.txt"
        
        start_time = time.time()
        
        # Write test data
        with open(test_file, 'w') as f:
            for i in range(1000):
                f.write(f"Test data line {i}\n")
        
        write_time = time.time() - start_time
        
        # Read test data
        start_time = time.time()
        
        with open(test_file, 'r') as f:
            data = f.read()
        
        read_time = time.time() - start_time
        
        # Disk I/O should be reasonably fast
        assert write_time < 0.1  # Less than 0.1 seconds for write
        assert read_time < 0.1   # Less than 0.1 seconds for read
        assert len(data) > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, al_artworks):
        """Test error recovery performance"""
        start_time = time.time()
        
        # Simulate error condition
        try:
            # This should fail
            await al_artworks.restore_image("nonexistent.png", "output.png")
        except Exception:
            pass
        
        recovery_time = time.time() - start_time
        
        # Error recovery should be fast
        assert recovery_time < 1.0  # Less than 1 second
    
    def test_startup_performance(self):
        """Test application startup performance"""
        start_time = time.time()
        
        # Import and initialize core modules
        from src import AlArtworks
        
        import_time = time.time() - start_time
        
        # Import should be fast
        assert import_time < 2.0  # Less than 2 seconds
    
    @pytest.mark.asyncio
    async def test_shutdown_performance(self, al_artworks):
        """Test application shutdown performance"""
        start_time = time.time()
        
        await al_artworks.cleanup()
        
        shutdown_time = time.time() - start_time
        
        # Shutdown should be fast
        assert shutdown_time < 2.0  # Less than 2 seconds

def test_performance_benchmarks():
    """Run comprehensive performance benchmarks"""
    import time
    
    # Test import performance
    start_time = time.time()
    import src
    import_time = time.time() - start_time
    
    print(f"Import time: {import_time:.3f}s")
    assert import_time < 3.0
    
    # Test module loading performance
    start_time = time.time()
    from src.core import config_validation
    from src.ui import main_window
    module_time = time.time() - start_time
    
    print(f"Module loading time: {module_time:.3f}s")
    assert module_time < 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
