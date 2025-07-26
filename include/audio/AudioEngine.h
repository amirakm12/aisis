#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

namespace Ultimate {
namespace Audio {

// Forward declarations
class AudioSource;
class AudioListener;
class AudioEffect;
class AudioBuffer;

// Audio formats
enum class AudioFormat {
    WAV,
    MP3,
    OGG,
    FLAC,
    AAC
};

// Audio quality
enum class AudioQuality {
    Low = 22050,
    Medium = 44100,
    High = 48000,
    Studio = 96000,
    Professional = 192000
};

// 3D audio modes
enum class Audio3DMode {
    Disabled,
    Simple,
    HRTF,
    Binaural,
    Surround
};

struct AudioStats {
    int activeSources = 0;
    int totalSources = 0;
    double cpuUsage = 0.0;
    size_t memoryUsage = 0;
    double latency = 0.0;
    int sampleRate = 44100;
    int bufferSize = 512;
};

class AudioEngine {
public:
    static AudioEngine& getInstance();
    
    // Initialization and cleanup
    bool initialize(AudioQuality quality = AudioQuality::High);
    void shutdown();
    
    // Master controls
    void setMasterVolume(float volume);
    float getMasterVolume() const;
    
    void setMasterMute(bool muted);
    bool isMasterMuted() const;
    
    // Audio sources
    std::shared_ptr<AudioSource> createAudioSource();
    std::shared_ptr<AudioSource> createAudioSource(const std::string& filepath);
    void removeAudioSource(std::shared_ptr<AudioSource> source);
    
    // Audio buffers
    std::shared_ptr<AudioBuffer> loadAudioFile(const std::string& filepath);
    std::shared_ptr<AudioBuffer> createAudioBuffer(const float* data, size_t samples, 
                                                  int channels, int sampleRate);
    
    // 3D Audio
    void set3DMode(Audio3DMode mode);
    Audio3DMode get3DMode() const;
    
    void setListenerPosition(float x, float y, float z);
    void setListenerVelocity(float x, float y, float z);
    void setListenerOrientation(float forwardX, float forwardY, float forwardZ,
                               float upX, float upY, float upZ);
    
    std::shared_ptr<AudioListener> getListener() const;
    
    // Audio effects
    void addGlobalEffect(const std::string& name, std::shared_ptr<AudioEffect> effect);
    void removeGlobalEffect(const std::string& name);
    void clearGlobalEffects();
    
    // Reverb
    void enableReverb(bool enable);
    bool isReverbEnabled() const;
    
    void setReverbPreset(const std::string& preset);
    std::vector<std::string> getAvailableReverbPresets() const;
    
    // Audio streaming
    void enableStreaming(bool enable);
    bool isStreamingEnabled() const;
    
    void setStreamingBufferSize(int size);
    int getStreamingBufferSize() const;
    
    // Quality and performance
    void setAudioQuality(AudioQuality quality);
    AudioQuality getAudioQuality() const;
    
    void setMaxSources(int maxSources);
    int getMaxSources() const;
    
    // Audio processing
    void enableDynamicRangeCompression(bool enable);
    bool isDynamicRangeCompressionEnabled() const;
    
    void enableNormalization(bool enable);
    bool isNormalizationEnabled() const;
    
    void enableSpatialAudio(bool enable);
    bool isSpatialAudioEnabled() const;
    
    // Audio analysis
    void enableSpectrumAnalysis(bool enable);
    bool isSpectrumAnalysisEnabled() const;
    
    std::vector<float> getFrequencySpectrum() const;
    float getAudioLevel() const;
    
    // Device management
    std::vector<std::string> getAvailableDevices() const;
    void setOutputDevice(const std::string& deviceName);
    std::string getCurrentOutputDevice() const;
    
    void setInputDevice(const std::string& deviceName);
    std::string getCurrentInputDevice() const;
    
    // Recording
    void startRecording(const std::string& filename = "");
    void stopRecording();
    bool isRecording() const;
    
    // Playback control
    void pauseAll();
    void resumeAll();
    void stopAll();
    
    // Volume mixing
    void setChannelVolume(const std::string& channel, float volume);
    float getChannelVolume(const std::string& channel) const;
    
    void muteChannel(const std::string& channel, bool muted);
    bool isChannelMuted(const std::string& channel) const;
    
    // Audio statistics
    const AudioStats& getAudioStats() const;
    void resetAudioStats();
    
    // Callbacks
    using AudioCallback = std::function<void()>;
    using AudioEventCallback = std::function<void(const std::string&)>;
    
    void setUpdateCallback(AudioCallback callback);
    void setErrorCallback(AudioEventCallback callback);
    void setDeviceChangeCallback(AudioEventCallback callback);
    
    // Advanced features
    void enableLowLatencyMode(bool enable);
    bool isLowLatencyModeEnabled() const;
    
    void enableExclusiveMode(bool enable);
    bool isExclusiveModeEnabled() const;
    
    void setBufferSize(int samples);
    int getBufferSize() const;

private:
    AudioEngine() = default;
    ~AudioEngine() = default;
    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;
    
    // Internal state
    bool m_initialized = false;
    AudioQuality m_audioQuality = AudioQuality::High;
    Audio3DMode m_3dMode = Audio3DMode::Simple;
    
    // Master controls
    float m_masterVolume = 1.0f;
    bool m_masterMuted = false;
    
    // Sources and buffers
    std::vector<std::shared_ptr<AudioSource>> m_audioSources;
    std::unordered_map<std::string, std::shared_ptr<AudioBuffer>> m_audioBuffers;
    
    // 3D Audio
    std::shared_ptr<AudioListener> m_listener;
    
    // Effects
    std::unordered_map<std::string, std::shared_ptr<AudioEffect>> m_globalEffects;
    bool m_reverbEnabled = false;
    std::string m_currentReverbPreset = "Hall";
    
    // Streaming
    bool m_streamingEnabled = true;
    int m_streamingBufferSize = 4096;
    
    // Processing
    bool m_dynamicRangeCompressionEnabled = false;
    bool m_normalizationEnabled = false;
    bool m_spatialAudioEnabled = true;
    
    // Analysis
    bool m_spectrumAnalysisEnabled = false;
    std::vector<float> m_frequencySpectrum;
    
    // Devices
    std::string m_currentOutputDevice;
    std::string m_currentInputDevice;
    
    // Recording
    bool m_recording = false;
    std::string m_recordingFilename;
    
    // Channel mixing
    std::unordered_map<std::string, float> m_channelVolumes;
    std::unordered_map<std::string, bool> m_channelMuted;
    
    // Performance
    int m_maxSources = 32;
    bool m_lowLatencyMode = false;
    bool m_exclusiveMode = false;
    int m_bufferSize = 512;
    
    // Statistics
    AudioStats m_audioStats;
    
    // Callbacks
    AudioCallback m_updateCallback;
    AudioEventCallback m_errorCallback;
    AudioEventCallback m_deviceChangeCallback;
    
    // Internal methods
    void updateAudioStats();
    void processAudioSources();
    void applyGlobalEffects();
    void updateListener();
};

} // namespace Audio
} // namespace Ultimate