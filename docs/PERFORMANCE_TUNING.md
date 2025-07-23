# AISIS Performance Tuning Guide

## Optimization Tips

### Hardware
- Use NVIDIA GPU with at least 8GB VRAM
- Minimum 16GB RAM

### Software
- Enable CUDA: Set `gpu_acceleration: true` in config.json
- Use quantized models for faster inference

### Configuration
- Set `max_processing_threads` to CPU cores
- Enable caching

### Monitoring
- Use `nvidia-smi` for GPU usage
- Profile with `cProfile`

For advanced tuning, see core/gpu_utils.py