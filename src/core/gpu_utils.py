"""
GPU Utilities for AISIS
Handles GPU detection, memory management, and optimization
"""

import torch
from loguru import logger


class GPUManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_info = self._get_gpu_info()

    def _get_gpu_info(self):
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)  # MB
            logger.info(f"Detected GPU: {name} with {total_mem} MB memory")
            return {"name": name, "total_memory_mb": total_mem}
        else:
            logger.info("No CUDA-compatible GPU detected, using CPU")
            return {"name": "CPU", "total_memory_mb": 0}

    def clear_cache(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")

    def get_memory_info(self):
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() // (1024**2)
            reserved = torch.cuda.memory_reserved() // (1024**2)
            free = reserved - allocated
            return {"allocated_mb": allocated, "reserved_mb": reserved, "free_mb": free}
        else:
            return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0}


gpu_manager = GPUManager()
