#pragma once

#include <portaudio.h>
#include <fftw3.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <string>

namespace aisis {

struct AudioFormat {
    int sampleRate{48000};
    int channels{2};
    int bitDepth{32}; // 16, 24, or 32
    int bufferSize{512};
};

struct AudioSample {
    std::vector<float> data;
    AudioFormat format;
    std::string name;
    float duration{0.0f};
    bool looping{false};
};

struct AudioTrack {
    uint32_t id;
    std::string name;
    std::vector<float> buffer;
    float volume{1.0f};
    float pan{0.0f}; // -1.0 (left) to 1.0 (right)
    bool muted{false};
    bool solo{false};
    std::vector<uint32_t> effectChain;
};

struct AudioEffect {
    enum Type {
        REVERB, DELAY, CHORUS, FLANGER, DISTORTION, COMPRESSOR,
        EQ, FILTER, PITCH_SHIFT, TIME_STRETCH, NOISE_GATE, LIMITER
    };
    
    Type type;
    std::unordered_map<std::string, float> parameters;
    bool enabled{true};
    bool bypass{false};
};

class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();
    
    // Initialization
    bool initialize(const AudioFormat& format = AudioFormat{});
    void shutdown();
    
    // Playback control
    bool start();
    bool stop();
    bool pause();
    bool resume();
    bool isPlaying() const { return m_playing; }
    
    // Sample management
    uint32_t loadSample(const std::string& filePath);
    uint32_t createSample(const std::vector<float>& data, const AudioFormat& format, const std::string& name);
    void deleteSample(uint32_t sampleId);
    bool playSample(uint32_t sampleId, float volume = 1.0f, float pitch = 1.0f);
    void stopSample(uint32_t sampleId);
    
    // Track management
    uint32_t createTrack(const std::string& name);
    void deleteTrack(uint32_t trackId);
    void setTrackVolume(uint32_t trackId, float volume);
    void setTrackPan(uint32_t trackId, float pan);
    void muteTrack(uint32_t trackId, bool muted);
    void soloTrack(uint32_t trackId, bool solo);
    
    // Recording
    bool startRecording(uint32_t trackId);
    bool stopRecording();
    bool isRecording() const { return m_recording; }
    void setRecordingFormat(const AudioFormat& format);
    
    // Effects processing
    uint32_t createEffect(AudioEffect::Type type);
    void deleteEffect(uint32_t effectId);
    void setEffectParameter(uint32_t effectId, const std::string& param, float value);
    void enableEffect(uint32_t effectId, bool enabled);
    void bypassEffect(uint32_t effectId, bool bypass);
    void addEffectToTrack(uint32_t trackId, uint32_t effectId);
    void removeEffectFromTrack(uint32_t trackId, uint32_t effectId);
    
    // Real-time analysis
    std::vector<float> getFFTSpectrum(int bins = 1024);
    float getRMSLevel(uint32_t trackId = 0); // 0 = master
    float getPeakLevel(uint32_t trackId = 0);
    std::vector<float> getWaveform(uint32_t trackId, int samples = 1024);
    
    // Master controls
    void setMasterVolume(float volume);
    float getMasterVolume() const { return m_masterVolume; }
    void enableMasterLimiter(bool enabled);
    void setMasterEQ(const std::vector<float>& bands); // 10-band EQ
    
    // Performance features
    void enableMultiThreadedProcessing(bool enabled);
    void setProcessingQuality(int quality); // 1-10 scale
    void enableZeroLatencyMode(bool enabled);
    void setBufferSize(int samples);
    
    // Advanced features
    void enableSpatialAudio(bool enabled);
    void set3DPosition(uint32_t trackId, float x, float y, float z);
    void setListenerPosition(float x, float y, float z);
    void enableDolbyAtmos(bool enabled);
    
    // MIDI support
    bool initializeMIDI();
    void processMIDIEvent(const std::vector<uint8_t>& midiData);
    void setMIDIMapping(uint32_t trackId, int midiChannel);
    
    // Synchronization
    void setTempo(float bpm);
    float getTempo() const { return m_tempo; }
    void setTimeSignature(int numerator, int denominator);
    void enableMetronome(bool enabled);
    void syncToExternalClock(bool enabled);
    
    // Export/Import
    bool exportToFile(const std::string& filePath, const AudioFormat& format);
    bool importFromFile(const std::string& filePath, uint32_t trackId);
    bool exportTrack(uint32_t trackId, const std::string& filePath);
    
    // Performance monitoring
    float getCPUUsage() const { return m_cpuUsage; }
    size_t getMemoryUsage() const { return m_memoryUsage; }
    float getLatency() const { return m_latency; }
    int getDroppedSamples() const { return m_droppedSamples; }
    
    // Real-time collaboration
    void enableNetworkStreaming(bool enabled);
    bool connectToSession(const std::string& sessionId);
    void broadcastAudio(bool enabled);
    
private:
    // PortAudio integration
    PaStream* m_stream{nullptr};
    AudioFormat m_format;
    std::atomic<bool> m_playing{false};
    std::atomic<bool> m_recording{false};
    
    // Audio data
    std::unordered_map<uint32_t, std::unique_ptr<AudioSample>> m_samples;
    std::unordered_map<uint32_t, std::unique_ptr<AudioTrack>> m_tracks;
    std::unordered_map<uint32_t, std::unique_ptr<AudioEffect>> m_effects;
    
    // Processing
    std::vector<float> m_inputBuffer;
    std::vector<float> m_outputBuffer;
    std::vector<float> m_mixBuffer;
    std::mutex m_bufferMutex;
    
    // FFT analysis
    fftwf_plan m_fftPlan;
    fftwf_complex* m_fftInput;
    fftwf_complex* m_fftOutput;
    std::vector<float> m_spectrum;
    std::mutex m_fftMutex;
    
    // Master controls
    std::atomic<float> m_masterVolume{1.0f};
    std::vector<float> m_masterEQ{10, 0.0f}; // 10-band EQ
    bool m_masterLimiterEnabled{true};
    
    // Performance settings
    bool m_multiThreadedProcessing{true};
    int m_processingQuality{8};
    bool m_zeroLatencyMode{false};
    std::atomic<float> m_cpuUsage{0.0f};
    std::atomic<size_t> m_memoryUsage{0};
    std::atomic<float> m_latency{0.0f};
    std::atomic<int> m_droppedSamples{0};
    
    // Timing and sync
    std::atomic<float> m_tempo{120.0f};
    int m_timeSignatureNum{4};
    int m_timeSignatureDen{4};
    bool m_metronomeEnabled{false};
    bool m_externalSyncEnabled{false};
    
    // 3D Audio
    bool m_spatialAudioEnabled{false};
    struct Position { float x, y, z; };
    std::unordered_map<uint32_t, Position> m_trackPositions;
    Position m_listenerPosition{0.0f, 0.0f, 0.0f};
    
    // Threading
    std::thread m_processingThread;
    std::thread m_analysisThread;
    std::atomic<bool> m_processingThreadRunning{false};
    std::atomic<bool> m_analysisThreadRunning{false};
    
    // ID generation
    std::atomic<uint32_t> m_nextSampleId{1};
    std::atomic<uint32_t> m_nextTrackId{1};
    std::atomic<uint32_t> m_nextEffectId{1};
    
    // Callback functions
    static int audioCallback(const void* inputBuffer, void* outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void* userData);
    
    // Processing methods
    void processAudio(const float* input, float* output, int frames);
    void mixTracks(float* output, int frames);
    void applyEffects(uint32_t trackId, float* buffer, int frames);
    void processEffect(const AudioEffect& effect, float* buffer, int frames);
    void updateAnalysis();
    void processingThreadFunction();
    void analysisThreadFunction();
    
    // Effect implementations
    void processReverb(float* buffer, int frames, const std::unordered_map<std::string, float>& params);
    void processDelay(float* buffer, int frames, const std::unordered_map<std::string, float>& params);
    void processCompressor(float* buffer, int frames, const std::unordered_map<std::string, float>& params);
    void processEQ(float* buffer, int frames, const std::unordered_map<std::string, float>& params);
    void processDistortion(float* buffer, int frames, const std::unordered_map<std::string, float>& params);
    
    // Utility methods
    void initializeFFT();
    void cleanupFFT();
    float calculateRMS(const float* buffer, int frames);
    float calculatePeak(const float* buffer, int frames);
    void applyGain(float* buffer, int frames, float gain);
    void applyPan(float* leftBuffer, float* rightBuffer, int frames, float pan);
};

} // namespace aisis