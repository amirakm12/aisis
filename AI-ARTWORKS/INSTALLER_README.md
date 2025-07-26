# AI-ARTWORK Enhanced Installer

## 🚀 Overview

The AI-ARTWORK Enhanced Installer is a comprehensive installation system with advanced features designed to provide a smooth, intelligent, and user-friendly installation experience.

## ✨ New Features

### 🧠 Smart Dependency Resolution
- **Automatic Detection**: Identifies missing, outdated, and conflicting packages
- **System Optimization**: Installs packages optimized for your system configuration  
- **Conditional Dependencies**: GPU-specific packages only installed when GPU is detected
- **Conflict Resolution**: Handles package conflicts intelligently
- **Fallback Support**: Graceful handling when optional packages aren't available

### 📊 Progress Tracking with ETA
- **Real-time Progress**: Live progress bars with percentage completion
- **ETA Calculation**: Intelligent estimation of remaining installation time
- **Step-by-step Updates**: Detailed progress for each installation phase
- **Download Speed Monitoring**: Track download speeds for model files
- **Visual Feedback**: Colorful terminal output with progress indicators

### 🔄 Rollback Support
- **Automatic Backup**: Creates backup before installation starts
- **Complete Rollback**: Uninstalls packages and removes files on failure
- **State Tracking**: Tracks all changes made during installation
- **Clean Recovery**: Restores system to pre-installation state
- **Error Handling**: Graceful failure recovery with detailed logging

### 🤖 Model Selection
- **Interactive Selection**: Choose which AI models to download
- **System-aware Recommendations**: Models recommended based on your hardware
- **Size Information**: Display model sizes and requirements
- **Category Organization**: Models organized by type (speech, language, image)
- **Compatibility Check**: Only shows models compatible with your system

### 🎮 GPU Detection
- **Automatic CUDA Detection**: Detects CUDA availability and version
- **Multi-GPU Support**: Handles systems with multiple GPUs
- **Memory Information**: Shows GPU memory capacity
- **Driver Version**: Displays GPU driver information
- **Compute Capability**: Shows GPU compute capability

### 🛣️ PATH Configuration
- **Automatic PATH Addition**: Adds installation to system PATH
- **Cross-platform Support**: Works on Windows, Linux, and macOS
- **User-level Changes**: Modifies user environment (no admin required)
- **Shell Integration**: Updates shell profiles on Unix systems
- **Registry Updates**: Updates Windows registry for PATH changes

### 🔗 Shortcuts Creation
- **Desktop Shortcuts**: Creates desktop shortcuts for easy access
- **Start Menu Integration**: Adds to Windows Start Menu or Linux applications
- **Cross-platform Icons**: Platform-appropriate shortcut formats
- **Automatic Detection**: Uses system-appropriate shortcut methods
- **Customizable**: Option to skip shortcut creation

### 🎯 First-Run Wizard
- **Interactive Setup**: Guided configuration on first run
- **System Analysis**: Analyzes your system for optimal settings
- **Preference Configuration**: Set theme, language, and performance options
- **GPU Configuration**: Configure GPU memory allocation and usage
- **Voice Settings**: Configure speech recognition and synthesis
- **Model Preferences**: Select and configure AI models
- **Performance Tuning**: Optimize settings for your hardware

## 📦 Installation Methods

### Method 1: Simple Installation
```bash
python install.py
```

### Method 2: Custom Path
```bash
python install.py --path /custom/installation/path
```

### Method 3: Advanced Options
```bash
python install.py --no-gpu --models whisper-tiny --dev
```

### Method 4: Quiet Installation
```bash
python install.py --quiet --auto-start
```

## 🛠️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--path PATH` | Installation directory | `~/AI-ARTWORK` |
| `--no-shortcuts` | Skip creating shortcuts | Creates shortcuts |
| `--no-path` | Don't add to system PATH | Adds to PATH |
| `--no-gpu` | Disable GPU support | Enables GPU |
| `--models MODEL1 MODEL2` | Specific models to install | `whisper-base llama-2-7b-chat` |
| `--dev` | Development mode | Production mode |
| `--auto-start` | Start after installation | Manual start |
| `--optional-deps` | Install optional dependencies | Core only |
| `--quiet` | Minimize output | Verbose output |

## 🎨 Model Selection

### Available Models

#### Speech Recognition (Whisper)
- **whisper-tiny** (39 MB) - Fast transcription, lower accuracy
- **whisper-base** (139 MB) - Balanced performance ⭐ *Recommended*
- **whisper-small** (244 MB) - Higher accuracy, slower

#### Language Models  
- **llama-2-7b-chat** (4 GB) - Conversational AI ⭐ *Recommended*
- **phi-2** (2 GB) - Reasoning and code generation

#### Image Generation
- **stable-diffusion-xl** (6 GB) - High-quality image generation ⭐ *Recommended*

### Model Requirements

| Model | RAM Required | GPU Required | Description |
|-------|-------------|--------------|-------------|
| whisper-tiny | 1 GB | No | Fast speech recognition |
| whisper-base | 2 GB | No | Balanced speech recognition |
| whisper-small | 3 GB | No | Accurate speech recognition |
| llama-2-7b-chat | 8 GB | Recommended | Conversational AI |
| phi-2 | 4 GB | No | Code and reasoning |
| stable-diffusion-xl | 12 GB | Yes | Image generation |

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 or higher
- **RAM**: 4 GB (8 GB recommended)
- **Storage**: 10 GB free space
- **Internet**: Required for model downloads

### Recommended Requirements
- **OS**: Windows 11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.10 or higher
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with 8 GB+ VRAM
- **Storage**: 50 GB free space (for all models)
- **Internet**: High-speed connection for faster downloads

## 🔧 Installation Process

### Phase 1: System Analysis
- Detect operating system and architecture
- Check Python version compatibility
- Analyze available RAM and storage
- Detect GPU capabilities and CUDA support
- Determine optimal configuration

### Phase 2: Dependency Resolution
- Scan for existing packages
- Identify missing dependencies
- Check for version conflicts
- Generate installation plan
- Prepare package list

### Phase 3: Installation
- Create backup of existing installation
- Install Python dependencies with progress tracking
- Set up directory structure
- Copy application files
- Configure system integration

### Phase 4: Model Download
- Download selected AI models
- Verify model integrity
- Organize models by category
- Create model index
- Update configuration

### Phase 5: System Integration
- Add installation to system PATH
- Create desktop and start menu shortcuts
- Configure file associations
- Set up auto-start options
- Register with system

### Phase 6: First-Run Configuration
- Launch interactive setup wizard
- Configure user preferences
- Optimize performance settings
- Test system integration
- Verify installation

## 🚨 Troubleshooting

### Common Issues

#### Installation Fails
1. Check Python version (3.8+ required)
2. Ensure sufficient disk space
3. Check internet connection
4. Run with administrator privileges if needed
5. Check the installation log for specific errors

#### GPU Not Detected
1. Verify NVIDIA GPU is present
2. Install latest GPU drivers
3. Install CUDA toolkit
4. Restart system after driver installation
5. Run `nvidia-smi` to verify GPU access

#### Models Won't Download
1. Check internet connection
2. Verify sufficient storage space
3. Check firewall settings
4. Try downloading individual models
5. Use VPN if regional restrictions exist

#### Shortcuts Not Created
1. Check user permissions
2. Verify desktop environment support
3. Try creating shortcuts manually
4. Check for antivirus interference
5. Run installer with elevated privileges

### Recovery Options

#### Rollback Installation
If installation fails, the system automatically attempts rollback:
```bash
# Manual rollback if needed
python -c "from scripts.enhanced_installer import RollbackManager; RollbackManager('/path/to/install').rollback()"
```

#### Clean Reinstall
```bash
# Remove existing installation
rm -rf ~/AI-ARTWORK
# Run fresh installation
python install.py
```

#### Reset Configuration
```bash
# Remove user configuration
rm ~/AI-ARTWORK/config/user_config.json
# Restart application for first-run wizard
python ~/AI-ARTWORK/launch.py
```

## 📝 Logs and Debugging

### Log Files
- **Installation Log**: `ai-artwork-install.log`
- **Application Log**: `~/AI-ARTWORK/logs/ai-artwork.log`
- **Error Log**: `~/AI-ARTWORK/logs/errors.log`

### Verbose Installation
```bash
python install.py --verbose
```

### Debug Mode
```bash
python install.py --debug
```

## 🔄 Updates and Maintenance

### Updating AI-ARTWORK
```bash
# Navigate to installation directory
cd ~/AI-ARTWORK
# Run update script
python scripts/update.py
```

### Updating Models
```bash
# Update all models
python scripts/download_models.py --update-all
# Update specific model
python scripts/download_models.py --update whisper-base
```

### Maintenance
```bash
# Clean cache
python scripts/maintenance.py --clean-cache
# Optimize models
python scripts/maintenance.py --optimize-models
# Check integrity
python scripts/maintenance.py --verify
```

## 🤝 Support

### Getting Help
- **Documentation**: Check the main README.md
- **Issues**: Report bugs on GitHub Issues
- **Community**: Join our Discord server
- **Email**: support@ai-artwork.com

### Before Reporting Issues
1. Check the troubleshooting section
2. Review installation logs
3. Try clean reinstallation
4. Test with minimal configuration
5. Gather system information

### System Information Script
```bash
python scripts/system_info.py
```

## 📄 License

This installer is part of the AI-ARTWORK project and is licensed under the same terms. See LICENSE file for details.

---

*Made with ❤️ by the AI-ARTWORK Team*