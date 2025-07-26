"""
Advanced Memory Management System with OOM Protection
Addresses critical memory requirements and prevents system crashes
"""

import gc
import psutil
import threading
import time
import warnings
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU memory management disabled")

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPUtil not available - GPU monitoring disabled")


@dataclass
class MemoryStats:
    """Memory statistics container"""
    total_ram: float
    available_ram: float
    used_ram: float
    ram_percent: float
    gpu_memory: Optional[Dict[str, float]] = None
    swap_usage: Optional[float] = None


class MemoryManager:
    """
    Comprehensive memory management system with OOM protection
    """
    
    def __init__(self, 
                 ram_threshold: float = 0.85,
                 gpu_threshold: float = 0.90,
                 cleanup_callbacks: Optional[List[Callable]] = None,
                 monitor_interval: float = 5.0):
        """
        Initialize memory manager
        
        Args:
            ram_threshold: RAM usage threshold (0.0-1.0) to trigger cleanup
            gpu_threshold: GPU memory threshold (0.0-1.0) to trigger cleanup
            cleanup_callbacks: List of functions to call during cleanup
            monitor_interval: Monitoring interval in seconds
        """
        self.ram_threshold = ram_threshold
        self.gpu_threshold = gpu_threshold
        self.cleanup_callbacks = cleanup_callbacks or []
        self.monitor_interval = monitor_interval
        
        self._monitoring = False
        self._monitor_thread = None
        self._memory_history = []
        self._max_history = 100
        
        # Emergency cleanup functions
        self._emergency_callbacks = []
        
        logger.info(f"Memory Manager initialized - RAM threshold: {ram_threshold*100:.1f}%, GPU threshold: {gpu_threshold*100:.1f}%")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # RAM statistics
        memory = psutil.virtual_memory()
        stats = MemoryStats(
            total_ram=memory.total / (1024**3),  # GB
            available_ram=memory.available / (1024**3),  # GB
            used_ram=memory.used / (1024**3),  # GB
            ram_percent=memory.percent / 100.0,
            swap_usage=psutil.swap_memory().percent / 100.0
        )
        
        # GPU statistics
        if GPU_AVAILABLE and TORCH_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_stats = {}
                for i, gpu in enumerate(gpus):
                    gpu_stats[f"gpu_{i}"] = {
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': gpu.memoryUtil,
                        'load': gpu.load
                    }
                stats.gpu_memory = gpu_stats
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")
        
        return stats
    
    def check_memory_pressure(self) -> Dict[str, bool]:
        """Check if memory pressure exists"""
        stats = self.get_memory_stats()
        pressure = {
            'ram_pressure': stats.ram_percent > self.ram_threshold,
            'gpu_pressure': False,
            'swap_pressure': stats.swap_usage and stats.swap_usage > 0.5
        }
        
        if stats.gpu_memory:
            for gpu_id, gpu_stats in stats.gpu_memory.items():
                if gpu_stats['memory_percent'] > self.gpu_threshold:
                    pressure['gpu_pressure'] = True
                    pressure[f'{gpu_id}_pressure'] = True
        
        return pressure
    
    def cleanup_memory(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform memory cleanup
        
        Args:
            force: Force aggressive cleanup
            
        Returns:
            Cleanup results
        """
        logger.info("Starting memory cleanup...")
        
        results = {
            'ram_freed': 0,
            'gpu_freed': 0,
            'callbacks_executed': 0,
            'torch_cache_cleared': False
        }
        
        # Get initial memory state
        initial_stats = self.get_memory_stats()
        
        # Execute cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
                results['callbacks_executed'] += 1
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
        
        # Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collector freed {collected} objects")
        
        # PyTorch specific cleanup
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    results['torch_cache_cleared'] = True
                    logger.debug("PyTorch CUDA cache cleared")
            except Exception as e:
                logger.error(f"PyTorch cleanup failed: {e}")
        
        # Force aggressive cleanup if needed
        if force:
            self._aggressive_cleanup()
        
        # Calculate freed memory
        final_stats = self.get_memory_stats()
        results['ram_freed'] = initial_stats.used_ram - final_stats.used_ram
        
        logger.info(f"Memory cleanup completed: {results}")
        return results
    
    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        logger.warning("Performing aggressive memory cleanup")
        
        # Execute emergency callbacks
        for callback in self._emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
        
        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()
        
        # Clear all possible caches
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except:
                pass
    
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        if self._monitoring:
            logger.warning("Memory monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                self._memory_history.append(stats)
                
                # Keep history bounded
                if len(self._memory_history) > self._max_history:
                    self._memory_history.pop(0)
                
                # Check for memory pressure
                pressure = self.check_memory_pressure()
                
                if pressure['ram_pressure']:
                    logger.warning(f"RAM pressure detected: {stats.ram_percent*100:.1f}%")
                    self.cleanup_memory()
                
                if pressure['gpu_pressure']:
                    logger.warning("GPU memory pressure detected")
                    self.cleanup_memory()
                
                # Check for critical memory situation
                if stats.ram_percent > 0.95:
                    logger.critical("CRITICAL: RAM usage > 95% - Emergency cleanup")
                    self.cleanup_memory(force=True)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
            
            time.sleep(self.monitor_interval)
    
    def add_cleanup_callback(self, callback: Callable):
        """Add a cleanup callback function"""
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Added cleanup callback: {callback.__name__}")
    
    def add_emergency_callback(self, callback: Callable):
        """Add an emergency cleanup callback"""
        self._emergency_callbacks.append(callback)
        logger.debug(f"Added emergency callback: {callback.__name__}")
    
    def get_memory_history(self) -> List[MemoryStats]:
        """Get memory usage history"""
        return self._memory_history.copy()
    
    def estimate_model_memory(self, model_size_gb: float) -> Dict[str, float]:
        """
        Estimate memory requirements for loading a model
        
        Args:
            model_size_gb: Model size in GB
            
        Returns:
            Memory estimates
        """
        # Conservative estimates including overhead
        estimates = {
            'model_memory': model_size_gb,
            'loading_overhead': model_size_gb * 0.5,  # 50% overhead during loading
            'inference_overhead': model_size_gb * 0.3,  # 30% overhead during inference
            'total_required': model_size_gb * 1.8  # Total with safety margin
        }
        
        return estimates
    
    def can_load_model(self, model_size_gb: float) -> Dict[str, Any]:
        """
        Check if a model can be safely loaded
        
        Args:
            model_size_gb: Model size in GB
            
        Returns:
            Loading feasibility analysis
        """
        stats = self.get_memory_stats()
        estimates = self.estimate_model_memory(model_size_gb)
        
        analysis = {
            'can_load': False,
            'available_ram': stats.available_ram,
            'required_memory': estimates['total_required'],
            'memory_deficit': max(0, estimates['total_required'] - stats.available_ram),
            'recommendations': []
        }
        
        if stats.available_ram >= estimates['total_required']:
            analysis['can_load'] = True
        else:
            analysis['recommendations'].append("Insufficient RAM for safe loading")
            if analysis['memory_deficit'] < 2.0:
                analysis['recommendations'].append("Try memory cleanup first")
            else:
                analysis['recommendations'].append("Consider model quantization or smaller variant")
        
        # GPU analysis
        if stats.gpu_memory:
            gpu_analysis = {}
            for gpu_id, gpu_stats in stats.gpu_memory.items():
                gpu_available = gpu_stats['memory_total'] - gpu_stats['memory_used']
                gpu_analysis[gpu_id] = {
                    'can_load': gpu_available >= estimates['total_required'] * 1024,  # Convert to MB
                    'available': gpu_available,
                    'required': estimates['total_required'] * 1024
                }
            analysis['gpu_analysis'] = gpu_analysis
        
        return analysis


# Global memory manager instance
memory_manager = MemoryManager()


def setup_memory_management(auto_start: bool = True) -> MemoryManager:
    """
    Setup global memory management
    
    Args:
        auto_start: Whether to start monitoring automatically
        
    Returns:
        Configured memory manager
    """
    if auto_start:
        memory_manager.start_monitoring()
    
    return memory_manager


def emergency_cleanup():
    """Emergency memory cleanup function"""
    logger.critical("EMERGENCY MEMORY CLEANUP TRIGGERED")
    memory_manager.cleanup_memory(force=True)


# Setup warning filters for memory issues
warnings.filterwarnings('error', category=ResourceWarning)