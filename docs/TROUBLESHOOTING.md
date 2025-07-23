# AISIS Troubleshooting Guide

## Common Issues

### GPU Not Detected
- Check `torch.cuda.is_available()`
- Install CUDA drivers
- Verify NVIDIA GPU

### Model Download Failures
- Check internet connection
- Run `python scripts/download_models.py` manually

### Voice System Errors
- Ensure microphone permissions
- Test with `test_core.py`

### Performance Issues
- Close background apps
- Use smaller models

Report persistent issues on GitHub.