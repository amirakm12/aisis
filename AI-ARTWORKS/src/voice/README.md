# Al-artworks Voice Modules

This folder contains all voice-related modules for Al-artworks, including:

## Voice Components

### Speech Recognition
- `faster_whisper_asr.py` - Real-time speech recognition using Faster Whisper
- `bark_tts.py` - Text-to-speech synthesis using Bark
- `voice_manager.py` - Voice input/output management

### Audio Processing
- Audio streaming and buffering
- Noise reduction and filtering
- Audio format conversion
- Real-time audio processing

### Voice Commands
- Natural language processing
- Command recognition and parsing
- Voice feedback and responses
- Multi-language support

## Key Features

### Real-time Speech Recognition
- Low-latency audio processing
- Continuous listening mode
- Background noise handling
- Multiple language support

### Text-to-Speech Synthesis
- Natural-sounding voice output
- Multiple voice options
- Emotion and tone control
- Real-time synthesis

### Voice Command Processing
- Natural language understanding
- Context-aware commands
- Voice feedback
- Command history

## Usage Examples

### Basic Voice Recognition
```python
from src.voice.faster_whisper_asr import WhisperASR

# Initialize speech recognition
asr = WhisperASR()
await asr.initialize()

# Start listening
async def on_speech(text):
    print(f"Recognized: {text}")

await asr.start_listening(on_speech)
```

### Text-to-Speech
```python
from src.voice.bark_tts import BarkTTS

# Initialize TTS
tts = BarkTTS()
await tts.initialize()

# Generate speech
audio = await tts.synthesize("Hello from Al-artworks!")
tts.play_audio(audio)
```

### Voice Manager
```python
from src.voice.voice_manager import VoiceManager

# Initialize voice manager
voice_mgr = VoiceManager()
await voice_mgr.initialize()

# Start voice interaction
await voice_mgr.start_voice_mode()

# Handle voice commands
async def on_command(text):
    print(f"Voice command: {text}")
    # Process command and respond
    response = await process_command(text)
    await voice_mgr.speak(response)

voice_mgr.command_received.connect(on_command)
```

## Configuration

### Voice Settings
```python
voice_config = {
    "recognition_model": "base",
    "language": "en",
    "sample_rate": 16000,
    "chunk_size": 1024,
    "noise_reduction": True,
    "voice_feedback": True
}
```

### Audio Settings
```python
audio_config = {
    "input_device": "default",
    "output_device": "default",
    "channels": 1,
    "format": "int16",
    "buffer_size": 4096
}
```

## Architecture

The voice system follows a modular design:

```
Voice System
├── Input Layer
│   ├── Audio Capture
│   ├── Noise Reduction
│   └── Buffering
├── Processing Layer
│   ├── Speech Recognition
│   ├── Command Parsing
│   └── Natural Language
├── Output Layer
│   ├── Text-to-Speech
│   ├── Audio Playback
│   └── Voice Feedback
└── Management Layer
    ├── Device Management
    ├── Configuration
    └── Error Handling
```

## Performance

### Optimization Tips
- Use appropriate audio chunk sizes
- Implement audio buffering
- Optimize model loading
- Cache frequently used audio

### Memory Management
- Release audio resources properly
- Clear audio buffers
- Monitor memory usage
- Implement garbage collection

## Dependencies

- **Faster Whisper** - Speech recognition
- **Bark** - Text-to-speech synthesis
- **PyAudio** - Audio I/O
- **NumPy** - Audio processing
- **SciPy** - Signal processing

## Development

### Adding New Voice Features

1. Create the module in the appropriate subdirectory
2. Implement proper async/await patterns
3. Add error handling and logging
4. Include unit tests
5. Update documentation

### Voice Command Processing

```python
class VoiceCommandProcessor:
    def __init__(self):
        self.commands = {}
        self.context = {}
    
    def register_command(self, pattern, handler):
        """Register a voice command pattern"""
        self.commands[pattern] = handler
    
    async def process_command(self, text):
        """Process voice command text"""
        for pattern, handler in self.commands.items():
            if pattern.match(text):
                return await handler(text, self.context)
        
        return "Command not recognized"
```

### Error Handling

```python
class VoiceError(Exception):
    """Base exception for voice-related errors"""
    pass

class AudioDeviceError(VoiceError):
    """Audio device not available"""
    pass

class RecognitionError(VoiceError):
    """Speech recognition failed"""
    pass

# Usage
try:
    await voice_mgr.start_listening()
except AudioDeviceError:
    print("No audio device available")
except RecognitionError:
    print("Speech recognition failed")
```

## Testing

### Voice Testing
```python
import pytest
from src.voice.voice_manager import VoiceManager

@pytest.mark.asyncio
async def test_voice_recognition():
    voice_mgr = VoiceManager()
    await voice_mgr.initialize()
    
    # Test speech recognition
    result = await voice_mgr.recognize_speech("test audio")
    assert result is not None
```

### Audio Testing
```python
def test_audio_processing():
    # Test audio format conversion
    audio_data = generate_test_audio()
    processed = process_audio(audio_data)
    assert len(processed) > 0
```

## Troubleshooting

### Common Issues

1. **No Audio Input**
   - Check microphone permissions
   - Verify audio device selection
   - Test with system audio tools

2. **Poor Recognition**
   - Reduce background noise
   - Speak clearly and slowly
   - Check microphone quality

3. **High Latency**
   - Optimize audio buffer size
   - Use faster hardware
   - Reduce processing complexity

### Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger('src.voice').setLevel(logging.DEBUG)

# Enable verbose output
voice_mgr.set_verbose(True)
``` 