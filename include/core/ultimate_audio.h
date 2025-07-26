#ifndef ULTIMATE_AUDIO_H
#define ULTIMATE_AUDIO_H

/**
 * @file ultimate_audio.h
 * @brief ULTIMATE Audio System
 * @version 1.0.0
 * @date 2024
 * 
 * Comprehensive audio processing and playback functionality for Windows.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "ultimate_types.h"
#include "ultimate_errors.h"

/* Audio formats */
typedef enum {
    ULTIMATE_AUDIO_FORMAT_PCM8 = 0,
    ULTIMATE_AUDIO_FORMAT_PCM16,
    ULTIMATE_AUDIO_FORMAT_PCM24,
    ULTIMATE_AUDIO_FORMAT_PCM32,
    ULTIMATE_AUDIO_FORMAT_FLOAT32,
    ULTIMATE_AUDIO_FORMAT_FLOAT64
} ultimate_audio_format_t;

/* Audio device types */
typedef enum {
    ULTIMATE_AUDIO_DEVICE_PLAYBACK = 0,
    ULTIMATE_AUDIO_DEVICE_CAPTURE,
    ULTIMATE_AUDIO_DEVICE_DUPLEX
} ultimate_audio_device_type_t;

/* Audio backend types */
typedef enum {
    ULTIMATE_AUDIO_BACKEND_WASAPI = 0,
    ULTIMATE_AUDIO_BACKEND_DIRECTSOUND,
    ULTIMATE_AUDIO_BACKEND_WAVEOUT,
    ULTIMATE_AUDIO_BACKEND_ASIO
} ultimate_audio_backend_t;

/* Audio stream states */
typedef enum {
    ULTIMATE_AUDIO_STREAM_STOPPED = 0,
    ULTIMATE_AUDIO_STREAM_PLAYING,
    ULTIMATE_AUDIO_STREAM_PAUSED,
    ULTIMATE_AUDIO_STREAM_RECORDING
} ultimate_audio_stream_state_t;

/* Audio effects */
typedef enum {
    ULTIMATE_AUDIO_EFFECT_REVERB = 0,
    ULTIMATE_AUDIO_EFFECT_ECHO,
    ULTIMATE_AUDIO_EFFECT_CHORUS,
    ULTIMATE_AUDIO_EFFECT_FLANGER,
    ULTIMATE_AUDIO_EFFECT_DISTORTION,
    ULTIMATE_AUDIO_EFFECT_COMPRESSOR,
    ULTIMATE_AUDIO_EFFECT_EQUALIZER,
    ULTIMATE_AUDIO_EFFECT_LIMITER
} ultimate_audio_effect_type_t;

/* Audio structures */
typedef struct {
    uint32_t sample_rate;
    uint32_t channels;
    ultimate_audio_format_t format;
    uint32_t buffer_size;
    uint32_t buffer_count;
} ultimate_audio_config_t;

typedef struct {
    char name[256];
    char description[512];
    uint32_t max_sample_rate;
    uint32_t min_sample_rate;
    uint32_t max_channels;
    bool is_default;
    ultimate_audio_device_type_t type;
} ultimate_audio_device_info_t;

typedef struct {
    float* data;
    uint32_t frames;
    uint32_t channels;
    uint32_t sample_rate;
    ultimate_audio_format_t format;
} ultimate_audio_buffer_t;

typedef struct {
    float volume;
    float pan;
    float pitch;
    bool mute;
    bool loop;
    uint32_t position;
    uint32_t length;
} ultimate_audio_source_info_t;

/* Handle types */
typedef void* ultimate_audio_device_handle_t;
typedef void* ultimate_audio_stream_handle_t;
typedef void* ultimate_audio_source_handle_t;
typedef void* ultimate_audio_effect_handle_t;
typedef void* ultimate_audio_mixer_handle_t;

/* Callback types */
typedef void (*ultimate_audio_callback_t)(ultimate_audio_buffer_t* input, ultimate_audio_buffer_t* output, void* user_data);
typedef void (*ultimate_audio_device_callback_t)(ultimate_audio_device_handle_t device, bool connected, void* user_data);

/* Audio system initialization */
ultimate_error_t ultimate_audio_init(ultimate_audio_backend_t backend);
ultimate_error_t ultimate_audio_shutdown(void);
ultimate_error_t ultimate_audio_set_master_volume(float volume);
ultimate_error_t ultimate_audio_get_master_volume(float* volume);
ultimate_error_t ultimate_audio_set_master_mute(bool mute);
ultimate_error_t ultimate_audio_get_master_mute(bool* mute);

/* Device management */
ultimate_error_t ultimate_audio_enumerate_devices(ultimate_audio_device_type_t type, ultimate_audio_device_info_t* devices, uint32_t* count);
ultimate_error_t ultimate_audio_get_default_device(ultimate_audio_device_type_t type, ultimate_audio_device_handle_t* device);
ultimate_error_t ultimate_audio_open_device(const char* device_name, ultimate_audio_device_type_t type, ultimate_audio_device_handle_t* device);
ultimate_error_t ultimate_audio_close_device(ultimate_audio_device_handle_t device);
ultimate_error_t ultimate_audio_get_device_info(ultimate_audio_device_handle_t device, ultimate_audio_device_info_t* info);
ultimate_error_t ultimate_audio_set_device_callback(ultimate_audio_device_handle_t device, ultimate_audio_device_callback_t callback, void* user_data);

/* Stream management */
ultimate_error_t ultimate_audio_stream_create(ultimate_audio_device_handle_t device, const ultimate_audio_config_t* config, ultimate_audio_stream_handle_t* stream);
ultimate_error_t ultimate_audio_stream_destroy(ultimate_audio_stream_handle_t stream);
ultimate_error_t ultimate_audio_stream_start(ultimate_audio_stream_handle_t stream);
ultimate_error_t ultimate_audio_stream_stop(ultimate_audio_stream_handle_t stream);
ultimate_error_t ultimate_audio_stream_pause(ultimate_audio_stream_handle_t stream);
ultimate_error_t ultimate_audio_stream_resume(ultimate_audio_stream_handle_t stream);
ultimate_error_t ultimate_audio_stream_get_state(ultimate_audio_stream_handle_t stream, ultimate_audio_stream_state_t* state);
ultimate_error_t ultimate_audio_stream_set_callback(ultimate_audio_stream_handle_t stream, ultimate_audio_callback_t callback, void* user_data);
ultimate_error_t ultimate_audio_stream_get_latency(ultimate_audio_stream_handle_t stream, uint32_t* latency_ms);
ultimate_error_t ultimate_audio_stream_get_position(ultimate_audio_stream_handle_t stream, uint64_t* position);

/* Audio source management */
ultimate_error_t ultimate_audio_source_create(ultimate_audio_source_handle_t* source);
ultimate_error_t ultimate_audio_source_destroy(ultimate_audio_source_handle_t source);
ultimate_error_t ultimate_audio_source_load_from_file(ultimate_audio_source_handle_t source, const char* filename);
ultimate_error_t ultimate_audio_source_load_from_memory(ultimate_audio_source_handle_t source, const void* data, uint32_t size);
ultimate_error_t ultimate_audio_source_play(ultimate_audio_source_handle_t source);
ultimate_error_t ultimate_audio_source_stop(ultimate_audio_source_handle_t source);
ultimate_error_t ultimate_audio_source_pause(ultimate_audio_source_handle_t source);
ultimate_error_t ultimate_audio_source_resume(ultimate_audio_source_handle_t source);
ultimate_error_t ultimate_audio_source_set_volume(ultimate_audio_source_handle_t source, float volume);
ultimate_error_t ultimate_audio_source_get_volume(ultimate_audio_source_handle_t source, float* volume);
ultimate_error_t ultimate_audio_source_set_pan(ultimate_audio_source_handle_t source, float pan);
ultimate_error_t ultimate_audio_source_get_pan(ultimate_audio_source_handle_t source, float* pan);
ultimate_error_t ultimate_audio_source_set_pitch(ultimate_audio_source_handle_t source, float pitch);
ultimate_error_t ultimate_audio_source_get_pitch(ultimate_audio_source_handle_t source, float* pitch);
ultimate_error_t ultimate_audio_source_set_loop(ultimate_audio_source_handle_t source, bool loop);
ultimate_error_t ultimate_audio_source_get_loop(ultimate_audio_source_handle_t source, bool* loop);
ultimate_error_t ultimate_audio_source_set_position(ultimate_audio_source_handle_t source, uint32_t position);
ultimate_error_t ultimate_audio_source_get_position(ultimate_audio_source_handle_t source, uint32_t* position);
ultimate_error_t ultimate_audio_source_get_length(ultimate_audio_source_handle_t source, uint32_t* length);
ultimate_error_t ultimate_audio_source_get_info(ultimate_audio_source_handle_t source, ultimate_audio_source_info_t* info);

/* Audio effects */
ultimate_error_t ultimate_audio_effect_create(ultimate_audio_effect_type_t type, ultimate_audio_effect_handle_t* effect);
ultimate_error_t ultimate_audio_effect_destroy(ultimate_audio_effect_handle_t effect);
ultimate_error_t ultimate_audio_effect_set_parameter(ultimate_audio_effect_handle_t effect, const char* name, float value);
ultimate_error_t ultimate_audio_effect_get_parameter(ultimate_audio_effect_handle_t effect, const char* name, float* value);
ultimate_error_t ultimate_audio_effect_enable(ultimate_audio_effect_handle_t effect, bool enable);
ultimate_error_t ultimate_audio_effect_is_enabled(ultimate_audio_effect_handle_t effect, bool* enabled);
ultimate_error_t ultimate_audio_effect_process(ultimate_audio_effect_handle_t effect, ultimate_audio_buffer_t* input, ultimate_audio_buffer_t* output);

/* Audio mixing */
ultimate_error_t ultimate_audio_mixer_create(const ultimate_audio_config_t* config, ultimate_audio_mixer_handle_t* mixer);
ultimate_error_t ultimate_audio_mixer_destroy(ultimate_audio_mixer_handle_t mixer);
ultimate_error_t ultimate_audio_mixer_add_source(ultimate_audio_mixer_handle_t mixer, ultimate_audio_source_handle_t source);
ultimate_error_t ultimate_audio_mixer_remove_source(ultimate_audio_mixer_handle_t mixer, ultimate_audio_source_handle_t source);
ultimate_error_t ultimate_audio_mixer_set_volume(ultimate_audio_mixer_handle_t mixer, float volume);
ultimate_error_t ultimate_audio_mixer_get_volume(ultimate_audio_mixer_handle_t mixer, float* volume);
ultimate_error_t ultimate_audio_mixer_mix(ultimate_audio_mixer_handle_t mixer, ultimate_audio_buffer_t* output);

/* Audio recording */
ultimate_error_t ultimate_audio_record_start(ultimate_audio_device_handle_t device, const char* filename, const ultimate_audio_config_t* config);
ultimate_error_t ultimate_audio_record_stop(void);
ultimate_error_t ultimate_audio_record_pause(void);
ultimate_error_t ultimate_audio_record_resume(void);
ultimate_error_t ultimate_audio_record_get_level(float* level);
ultimate_error_t ultimate_audio_record_set_gain(float gain);
ultimate_error_t ultimate_audio_record_get_gain(float* gain);

/* Audio format conversion */
ultimate_error_t ultimate_audio_convert_format(const ultimate_audio_buffer_t* input, ultimate_audio_buffer_t* output, ultimate_audio_format_t target_format);
ultimate_error_t ultimate_audio_resample(const ultimate_audio_buffer_t* input, ultimate_audio_buffer_t* output, uint32_t target_sample_rate);
ultimate_error_t ultimate_audio_mix_channels(const ultimate_audio_buffer_t* input, ultimate_audio_buffer_t* output, uint32_t target_channels);

/* Audio file I/O */
ultimate_error_t ultimate_audio_file_load(const char* filename, ultimate_audio_buffer_t* buffer);
ultimate_error_t ultimate_audio_file_save(const char* filename, const ultimate_audio_buffer_t* buffer);
ultimate_error_t ultimate_audio_file_get_info(const char* filename, ultimate_audio_config_t* config);
ultimate_error_t ultimate_audio_file_supports_format(const char* filename, bool* supported);

/* Audio analysis */
ultimate_error_t ultimate_audio_analyze_spectrum(const ultimate_audio_buffer_t* buffer, float* spectrum, uint32_t bins);
ultimate_error_t ultimate_audio_analyze_peak(const ultimate_audio_buffer_t* buffer, float* peak);
ultimate_error_t ultimate_audio_analyze_rms(const ultimate_audio_buffer_t* buffer, float* rms);
ultimate_error_t ultimate_audio_detect_silence(const ultimate_audio_buffer_t* buffer, float threshold, bool* is_silent);
ultimate_error_t ultimate_audio_detect_clipping(const ultimate_audio_buffer_t* buffer, bool* is_clipping);

/* 3D audio */
ultimate_error_t ultimate_audio_3d_init(void);
ultimate_error_t ultimate_audio_3d_shutdown(void);
ultimate_error_t ultimate_audio_3d_set_listener_position(const ultimate_vector3_t* position);
ultimate_error_t ultimate_audio_3d_set_listener_orientation(const ultimate_vector3_t* forward, const ultimate_vector3_t* up);
ultimate_error_t ultimate_audio_3d_set_listener_velocity(const ultimate_vector3_t* velocity);
ultimate_error_t ultimate_audio_3d_set_source_position(ultimate_audio_source_handle_t source, const ultimate_vector3_t* position);
ultimate_error_t ultimate_audio_3d_set_source_velocity(ultimate_audio_source_handle_t source, const ultimate_vector3_t* velocity);
ultimate_error_t ultimate_audio_3d_set_source_direction(ultimate_audio_source_handle_t source, const ultimate_vector3_t* direction);
ultimate_error_t ultimate_audio_3d_set_distance_model(int model);
ultimate_error_t ultimate_audio_3d_set_doppler_factor(float factor);

/* Audio synthesis */
ultimate_error_t ultimate_audio_generate_sine(ultimate_audio_buffer_t* buffer, float frequency, float amplitude, float duration);
ultimate_error_t ultimate_audio_generate_square(ultimate_audio_buffer_t* buffer, float frequency, float amplitude, float duration);
ultimate_error_t ultimate_audio_generate_sawtooth(ultimate_audio_buffer_t* buffer, float frequency, float amplitude, float duration);
ultimate_error_t ultimate_audio_generate_triangle(ultimate_audio_buffer_t* buffer, float frequency, float amplitude, float duration);
ultimate_error_t ultimate_audio_generate_noise_white(ultimate_audio_buffer_t* buffer, float amplitude, float duration);
ultimate_error_t ultimate_audio_generate_noise_pink(ultimate_audio_buffer_t* buffer, float amplitude, float duration);
ultimate_error_t ultimate_audio_generate_silence(ultimate_audio_buffer_t* buffer, float duration);

/* Audio utilities */
ultimate_error_t ultimate_audio_buffer_create(uint32_t frames, uint32_t channels, ultimate_audio_format_t format, ultimate_audio_buffer_t* buffer);
ultimate_error_t ultimate_audio_buffer_destroy(ultimate_audio_buffer_t* buffer);
ultimate_error_t ultimate_audio_buffer_copy(const ultimate_audio_buffer_t* src, ultimate_audio_buffer_t* dst);
ultimate_error_t ultimate_audio_buffer_clear(ultimate_audio_buffer_t* buffer);
ultimate_error_t ultimate_audio_buffer_resize(ultimate_audio_buffer_t* buffer, uint32_t frames);
ultimate_error_t ultimate_audio_buffer_append(ultimate_audio_buffer_t* buffer, const ultimate_audio_buffer_t* source);
ultimate_error_t ultimate_audio_buffer_normalize(ultimate_audio_buffer_t* buffer);
ultimate_error_t ultimate_audio_buffer_fade_in(ultimate_audio_buffer_t* buffer, float duration);
ultimate_error_t ultimate_audio_buffer_fade_out(ultimate_audio_buffer_t* buffer, float duration);
ultimate_error_t ultimate_audio_buffer_reverse(ultimate_audio_buffer_t* buffer);

/* Performance monitoring */
ultimate_error_t ultimate_audio_get_cpu_usage(float* usage);
ultimate_error_t ultimate_audio_get_memory_usage(uint64_t* usage);
ultimate_error_t ultimate_audio_get_active_sources(uint32_t* count);
ultimate_error_t ultimate_audio_get_sample_rate(uint32_t* sample_rate);
ultimate_error_t ultimate_audio_get_buffer_size(uint32_t* buffer_size);

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_AUDIO_H */