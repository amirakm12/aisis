#ifndef ULTIMATE_HARDWARE_H
#define ULTIMATE_HARDWARE_H

/**
 * @file ultimate_hardware.h
 * @brief ULTIMATE Windows Hardware Abstraction Layer
 * @version 1.0.0
 * @date 2024
 * 
 * Windows-specific hardware abstraction layer for the ULTIMATE system.
 * Provides unified interface for Windows API integration.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "ultimate_types.h"
#include "ultimate_errors.h"

/* Windows-specific hardware types */
typedef enum {
    ULTIMATE_WINDOW_STYLE_NORMAL = 0,
    ULTIMATE_WINDOW_STYLE_BORDERLESS,
    ULTIMATE_WINDOW_STYLE_FULLSCREEN,
    ULTIMATE_WINDOW_STYLE_TOOL_WINDOW
} ultimate_window_style_t;

typedef enum {
    ULTIMATE_INPUT_TYPE_KEYBOARD = 0,
    ULTIMATE_INPUT_TYPE_MOUSE,
    ULTIMATE_INPUT_TYPE_TOUCH,
    ULTIMATE_INPUT_TYPE_GAMEPAD
} ultimate_input_type_t;

typedef enum {
    ULTIMATE_AUDIO_FORMAT_PCM = 0,
    ULTIMATE_AUDIO_FORMAT_MP3,
    ULTIMATE_AUDIO_FORMAT_WAV,
    ULTIMATE_AUDIO_FORMAT_OGG
} ultimate_audio_format_t;

/* Window management */
ultimate_error_t ultimate_window_create(const ultimate_window_config_t* config,
                                       ultimate_window_handle_t* window);
ultimate_error_t ultimate_window_destroy(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_show(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_hide(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_set_title(ultimate_window_handle_t window, const char* title);
ultimate_error_t ultimate_window_set_size(ultimate_window_handle_t window, uint32_t width, uint32_t height);
ultimate_error_t ultimate_window_set_position(ultimate_window_handle_t window, int32_t x, int32_t y);

/* Window information */
ultimate_error_t ultimate_window_get_size(ultimate_window_handle_t window, uint32_t* width, uint32_t* height);
ultimate_error_t ultimate_window_get_position(ultimate_window_handle_t window, int32_t* x, int32_t* y);
bool ultimate_window_is_visible(ultimate_window_handle_t window);
bool ultimate_window_has_focus(ultimate_window_handle_t window);

/* Input handling */
typedef struct {
    ultimate_input_type_t type;
    uint32_t key_code;
    uint32_t modifiers;
    int32_t mouse_x;
    int32_t mouse_y;
    uint32_t timestamp;
} ultimate_input_event_t;

typedef void (*ultimate_input_callback_t)(ultimate_window_handle_t window, 
                                         const ultimate_input_event_t* event,
                                         void* user_data);

ultimate_error_t ultimate_input_register_callback(ultimate_window_handle_t window,
                                                 ultimate_input_callback_t callback,
                                                 void* user_data);

ultimate_error_t ultimate_input_get_key_state(uint32_t key_code, bool* is_pressed);
ultimate_error_t ultimate_input_get_mouse_position(int32_t* x, int32_t* y);

/* Graphics and rendering */
typedef struct ultimate_graphics_context* ultimate_graphics_handle_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t color_depth;
    bool vsync_enabled;
    bool fullscreen;
} ultimate_graphics_config_t;

ultimate_error_t ultimate_graphics_init(ultimate_window_handle_t window,
                                       const ultimate_graphics_config_t* config,
                                       ultimate_graphics_handle_t* graphics);
ultimate_error_t ultimate_graphics_deinit(ultimate_graphics_handle_t graphics);

ultimate_error_t ultimate_graphics_begin_frame(ultimate_graphics_handle_t graphics);
ultimate_error_t ultimate_graphics_end_frame(ultimate_graphics_handle_t graphics);
ultimate_error_t ultimate_graphics_clear(ultimate_graphics_handle_t graphics, uint32_t color);
ultimate_error_t ultimate_graphics_present(ultimate_graphics_handle_t graphics);

/* Audio system */
typedef struct ultimate_audio_context* ultimate_audio_handle_t;

typedef struct {
    uint32_t sample_rate;
    uint32_t channels;
    uint32_t bits_per_sample;
    ultimate_audio_format_t format;
} ultimate_audio_config_t;

ultimate_error_t ultimate_audio_init(const ultimate_audio_config_t* config,
                                    ultimate_audio_handle_t* audio);
ultimate_error_t ultimate_audio_deinit(ultimate_audio_handle_t audio);

ultimate_error_t ultimate_audio_play(ultimate_audio_handle_t audio,
                                    const void* data,
                                    size_t data_size);
ultimate_error_t ultimate_audio_stop(ultimate_audio_handle_t audio);
ultimate_error_t ultimate_audio_set_volume(ultimate_audio_handle_t audio, float volume);

/* File system */
typedef struct ultimate_file* ultimate_file_handle_t;

typedef enum {
    ULTIMATE_FILE_MODE_READ = 0,
    ULTIMATE_FILE_MODE_WRITE,
    ULTIMATE_FILE_MODE_APPEND,
    ULTIMATE_FILE_MODE_READ_WRITE
} ultimate_file_mode_t;

ultimate_error_t ultimate_file_open(const char* path,
                                   ultimate_file_mode_t mode,
                                   ultimate_file_handle_t* file);
ultimate_error_t ultimate_file_close(ultimate_file_handle_t file);
ultimate_error_t ultimate_file_read(ultimate_file_handle_t file,
                                   void* buffer,
                                   size_t size,
                                   size_t* bytes_read);
ultimate_error_t ultimate_file_write(ultimate_file_handle_t file,
                                    const void* buffer,
                                    size_t size,
                                    size_t* bytes_written);
ultimate_error_t ultimate_file_seek(ultimate_file_handle_t file, int64_t offset, int whence);
ultimate_error_t ultimate_file_get_size(ultimate_file_handle_t file, size_t* size);

/* Network communication */
typedef struct ultimate_socket* ultimate_socket_handle_t;

typedef enum {
    ULTIMATE_SOCKET_TYPE_TCP = 0,
    ULTIMATE_SOCKET_TYPE_UDP
} ultimate_socket_type_t;

ultimate_error_t ultimate_socket_create(ultimate_socket_type_t type,
                                       ultimate_socket_handle_t* socket);
ultimate_error_t ultimate_socket_destroy(ultimate_socket_handle_t socket);
ultimate_error_t ultimate_socket_connect(ultimate_socket_handle_t socket,
                                        const char* address,
                                        uint16_t port);
ultimate_error_t ultimate_socket_bind(ultimate_socket_handle_t socket,
                                     const char* address,
                                     uint16_t port);
ultimate_error_t ultimate_socket_listen(ultimate_socket_handle_t socket, int backlog);
ultimate_error_t ultimate_socket_accept(ultimate_socket_handle_t socket,
                                       ultimate_socket_handle_t* client_socket);
ultimate_error_t ultimate_socket_send(ultimate_socket_handle_t socket,
                                     const void* data,
                                     size_t size,
                                     size_t* bytes_sent);
ultimate_error_t ultimate_socket_receive(ultimate_socket_handle_t socket,
                                        void* buffer,
                                        size_t size,
                                        size_t* bytes_received);

/* System information */
typedef struct {
    uint32_t cpu_count;
    uint64_t total_memory;
    uint64_t available_memory;
    const char* os_version;
    const char* computer_name;
    const char* user_name;
} ultimate_system_info_t;

ultimate_error_t ultimate_system_get_hardware_info(ultimate_system_info_t* info);

/* Registry access */
ultimate_error_t ultimate_registry_read_string(const char* key_path,
                                              const char* value_name,
                                              char* buffer,
                                              size_t buffer_size);
ultimate_error_t ultimate_registry_write_string(const char* key_path,
                                              const char* value_name,
                                              const char* value);
ultimate_error_t ultimate_registry_read_dword(const char* key_path,
                                             const char* value_name,
                                             uint32_t* value);
ultimate_error_t ultimate_registry_write_dword(const char* key_path,
                                              const char* value_name,
                                              uint32_t value);

/* Process and thread management */
typedef struct ultimate_process* ultimate_process_handle_t;

ultimate_error_t ultimate_process_create(const char* command_line,
                                        ultimate_process_handle_t* process);
ultimate_error_t ultimate_process_destroy(ultimate_process_handle_t process);
ultimate_error_t ultimate_process_wait(ultimate_process_handle_t process, uint32_t timeout_ms);
ultimate_error_t ultimate_process_get_exit_code(ultimate_process_handle_t process, int* exit_code);

/* Service management */
ultimate_error_t ultimate_service_install(const char* service_name,
                                         const char* display_name,
                                         const char* binary_path);
ultimate_error_t ultimate_service_uninstall(const char* service_name);
ultimate_error_t ultimate_service_start(const char* service_name);
ultimate_error_t ultimate_service_stop(const char* service_name);

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_HARDWARE_H */ 