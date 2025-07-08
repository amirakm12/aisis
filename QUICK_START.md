# 🚀 AISIS QUICK START GUIDE

## ⚡ **5-Minute Setup**

### **1. Install Dependencies**
```bash
# Navigate to project directory
cd aisis

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### **2. Download AI Models**
```bash
# Download essential models (this may take 10-30 minutes)
python scripts/download_models.py

# Check downloaded models
ls -la models/
```

### **3. Test Core Functionality**
```bash
# Run basic tests
python -m pytest tests/test_agents.py -v

# Test configuration
python -c "from src.core.config import config; print('✓ Config loaded')"
```

### **4. Launch Application**
```bash
# Start AISIS
python main.py

# Or use alternative launcher
python launch.py
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. CUDA/GPU Issues**
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If no GPU, models will use CPU (slower but functional)
```

#### **2. Memory Issues**
```bash
# Check available memory
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().available / 1e9:.1f} GB')"

# Reduce model size in config.json if needed
```

#### **3. Missing Dependencies**
```bash
# Reinstall with verbose output
pip install -r requirements.txt -v

# Check specific package
pip show torch
```

#### **4. Model Download Failures**
```bash
# Manual download retry
python scripts/download_models.py --retry-failed

# Check internet connection
curl -I https://huggingface.co
```

## 📋 **Quick Test Commands**

### **Test Individual Components**
```bash
# Test image restoration
python -c "
from src.agents.image_restoration import ImageRestorationAgent
agent = ImageRestorationAgent()
print('✓ Image restoration agent ready')
"

# Test voice processing
python -c "
from src.core.voice.faster_whisper_asr import FasterWhisperASR
asr = FasterWhisperASR()
print('✓ Voice processing ready')
"

# Test UI
python -c "
from src.ui.main_window import MainWindow
print('✓ UI components ready')
"
```

### **Test Full Pipeline**
```bash
# Run integration test
python -m pytest tests/test_integration.py -v

# Test with sample data
python -c "
import asyncio
from src.agents.orchestrator import OrchestratorAgent
async def test():
    agent = OrchestratorAgent()
    await agent.initialize()
    print('✓ Orchestrator ready')
asyncio.run(test())
"
```

## 🎯 **First Use**

### **1. Basic Image Processing**
1. Launch AISIS
2. Load an image
3. Select "Auto Enhancement"
4. Wait for processing
5. Save result

### **2. Voice Commands**
1. Click microphone button
2. Speak command
3. Wait for transcription
4. View results

### **3. AI Generation**
1. Enter text prompt
2. Select generation type
3. Adjust parameters
4. Generate and save

## 📊 **Performance Tips**

### **For Better Performance**
```bash
# Use GPU if available
export CUDA_VISIBLE_DEVICES=0

# Increase memory limit
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use smaller models for testing
# Edit config.json: set "model_size": "small"
```

### **For Development**
```bash
# Enable debug mode
export AISIS_DEBUG=1

# Verbose logging
export AISIS_LOG_LEVEL=DEBUG

# Test mode
python main.py --test-mode
```

## 🚨 **Emergency Commands**

### **Reset Everything**
```bash
# Clear all caches
rm -rf .aisis_cache/
rm -rf models/
rm -rf __pycache__/

# Reinstall from scratch
pip uninstall -y aisis
pip install -e .
```

### **Quick Diagnostics**
```bash
# System check
python scripts/setup_environment.py --diagnose

# Model verification
python scripts/download_models.py --verify

# Performance test
python -m pytest tests/test_performance.py -v
```

## 📞 **Get Help**

### **Documentation**
- [Full Documentation](README_ENHANCED.md)
- [API Reference](docs/API_REFERENCE.md)
- [User Manual](docs/USER_MANUAL.md)

### **Support**
- [GitHub Issues](https://github.com/your-repo/aisis/issues)
- [Discussions](https://github.com/your-repo/aisis/discussions)
- [Wiki](https://github.com/your-repo/aisis/wiki)

### **Community**
- Join our Discord server
- Follow on Twitter
- Subscribe to newsletter

---

**Need immediate help?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or open a GitHub issue. 