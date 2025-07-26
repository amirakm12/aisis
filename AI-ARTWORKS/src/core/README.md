# Al-artworks Core Modules

This folder contains the core infrastructure for Al-artworks, including configuration, GPU utilities, LLM management, and voice management.

## Core Components

### Configuration Management
- `config.py` - Main configuration system
- `config_validation.py` - Configuration validation using Pydantic
- `logging_setup.py` - Centralized logging configuration

### GPU and Hardware Management
- `gpu_utils.py` - GPU detection and management utilities
- `advanced_local_models.py` - Local model management and optimization

### AI and ML Infrastructure
- `llm_manager.py` - Large language model management
- `voice_manager.py` - Voice input/output processing
- `voice/` - Voice processing modules

### System Integration
- `integration.py` - External system integration
- `device.py` - Device-specific adapters

## Key Features

### Configuration System
- Type-safe configuration using Pydantic
- Environment variable support
- Configuration validation and error handling
- Hot-reload capability

### GPU Management
- Automatic CUDA detection
- Memory management and optimization
- Multi-GPU support
- Fallback to CPU when needed

### Voice Processing
- Real-time speech recognition
- Text-to-speech synthesis
- Voice command processing
- Audio streaming and processing

### Logging and Monitoring
- Structured logging with loguru
- Performance monitoring
- Error tracking and reporting
- Debug and trace logging

## Usage Examples

### Configuration
```python
from src.core.config_validation import AlArtworksConfig

# Load configuration
config = AlArtworksConfig(
    app_name="Al-artworks",
    device="auto",
    quality="high"
)

# Validate configuration
config.validate()
```

### GPU Management
```python
from src.core.gpu_utils import gpu_manager

# Initialize GPU
gpu_manager.initialize()

# Check GPU availability
if gpu_manager.is_available():
    device = gpu_manager.get_device()
    print(f"Using GPU: {device}")
```

### Voice Processing
```python
from src.core.voice_manager import voice_manager

# Initialize voice processing
await voice_manager.initialize()

# Start voice input
await voice_manager.start_listening()

# Process voice commands
async def on_command(text):
    print(f"Voice command: {text}")
```

### Logging
```python
from src.core.logging_setup import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", log_file="logs/al_artworks.log")

# Get logger
logger = get_logger(__name__)
logger.info("Application started")
```

## Architecture

The core modules provide a solid foundation for the Al-artworks system:

1. **Configuration Layer** - Manages all system settings and parameters
2. **Hardware Layer** - Handles GPU, CPU, and device management
3. **AI Layer** - Provides LLM and voice processing capabilities
4. **Integration Layer** - Connects to external systems and services
5. **Monitoring Layer** - Tracks performance and system health

## Dependencies

- **Pydantic** - Configuration validation
- **Loguru** - Advanced logging
- **PyTorch** - GPU operations
- **Transformers** - LLM management
- **SpeechRecognition** - Voice processing

## Development

### Adding New Core Components

1. Create the module in the appropriate subdirectory
2. Add proper type hints and documentation
3. Include unit tests in `tests/test_core.py`
4. Update this README with usage examples

### Configuration Schema

When adding new configuration options:

1. Update `AlArtworksConfig` in `config_validation.py`
2. Add validation rules if needed
3. Include default values and descriptions
4. Update documentation

### Performance Considerations

- Use async/await for I/O operations
- Implement proper resource cleanup
- Monitor memory usage
- Cache frequently used data
- Use connection pooling for external services 