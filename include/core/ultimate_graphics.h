#ifndef ULTIMATE_GRAPHICS_H
#define ULTIMATE_GRAPHICS_H

/**
 * @file ultimate_graphics.h
 * @brief ULTIMATE Graphics and GUI System
 * @version 1.0.0
 * @date 2024
 * 
 * Comprehensive graphics, GUI, and rendering functionality for Windows.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "ultimate_types.h"
#include "ultimate_errors.h"

/* Graphics context types */
typedef enum {
    ULTIMATE_GRAPHICS_CONTEXT_DIRECTX11 = 0,
    ULTIMATE_GRAPHICS_CONTEXT_DIRECTX12,
    ULTIMATE_GRAPHICS_CONTEXT_OPENGL,
    ULTIMATE_GRAPHICS_CONTEXT_VULKAN,
    ULTIMATE_GRAPHICS_CONTEXT_SOFTWARE
} ultimate_graphics_context_type_t;

/* Window styles */
typedef enum {
    ULTIMATE_WINDOW_STYLE_NORMAL = 0,
    ULTIMATE_WINDOW_STYLE_BORDERLESS,
    ULTIMATE_WINDOW_STYLE_DIALOG,
    ULTIMATE_WINDOW_STYLE_TOOL_WINDOW,
    ULTIMATE_WINDOW_STYLE_POPUP
} ultimate_window_style_t;

/* Color formats */
typedef enum {
    ULTIMATE_COLOR_FORMAT_RGB = 0,
    ULTIMATE_COLOR_FORMAT_RGBA,
    ULTIMATE_COLOR_FORMAT_BGR,
    ULTIMATE_COLOR_FORMAT_BGRA,
    ULTIMATE_COLOR_FORMAT_HSV,
    ULTIMATE_COLOR_FORMAT_HSL
} ultimate_color_format_t;

/* Texture formats */
typedef enum {
    ULTIMATE_TEXTURE_FORMAT_RGB8 = 0,
    ULTIMATE_TEXTURE_FORMAT_RGBA8,
    ULTIMATE_TEXTURE_FORMAT_RGB16,
    ULTIMATE_TEXTURE_FORMAT_RGBA16,
    ULTIMATE_TEXTURE_FORMAT_RGB32F,
    ULTIMATE_TEXTURE_FORMAT_RGBA32F,
    ULTIMATE_TEXTURE_FORMAT_DXT1,
    ULTIMATE_TEXTURE_FORMAT_DXT5
} ultimate_texture_format_t;

/* Render targets */
typedef enum {
    ULTIMATE_RENDER_TARGET_BACKBUFFER = 0,
    ULTIMATE_RENDER_TARGET_TEXTURE,
    ULTIMATE_RENDER_TARGET_DEPTH_BUFFER,
    ULTIMATE_RENDER_TARGET_SHADOW_MAP
} ultimate_render_target_type_t;

/* Primitive types */
typedef enum {
    ULTIMATE_PRIMITIVE_POINTS = 0,
    ULTIMATE_PRIMITIVE_LINES,
    ULTIMATE_PRIMITIVE_LINE_STRIP,
    ULTIMATE_PRIMITIVE_TRIANGLES,
    ULTIMATE_PRIMITIVE_TRIANGLE_STRIP,
    ULTIMATE_PRIMITIVE_TRIANGLE_FAN,
    ULTIMATE_PRIMITIVE_QUADS
} ultimate_primitive_type_t;

/* Blend modes */
typedef enum {
    ULTIMATE_BLEND_NONE = 0,
    ULTIMATE_BLEND_ALPHA,
    ULTIMATE_BLEND_ADDITIVE,
    ULTIMATE_BLEND_MULTIPLY,
    ULTIMATE_BLEND_SCREEN,
    ULTIMATE_BLEND_OVERLAY
} ultimate_blend_mode_t;

/* Graphics structures */
typedef struct {
    float x, y, z, w;
} ultimate_vector4_t;

typedef struct {
    float x, y, z;
} ultimate_vector3_t;

typedef struct {
    float x, y;
} ultimate_vector2_t;

typedef struct {
    float r, g, b, a;
} ultimate_color_t;

typedef struct {
    int32_t x, y;
    uint32_t width, height;
} ultimate_rect_t;

typedef struct {
    ultimate_vector3_t position;
    ultimate_vector3_t normal;
    ultimate_vector2_t texcoord;
    ultimate_color_t color;
} ultimate_vertex_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    ultimate_texture_format_t format;
    void* data;
    uint32_t data_size;
    uint32_t mip_levels;
} ultimate_texture_desc_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    bool fullscreen;
    bool vsync;
    uint32_t samples;
    ultimate_graphics_context_type_t context_type;
    const char* title;
} ultimate_graphics_config_t;

/* Handle types */
typedef void* ultimate_graphics_context_t;
typedef void* ultimate_texture_handle_t;
typedef void* ultimate_shader_handle_t;
typedef void* ultimate_buffer_handle_t;
typedef void* ultimate_render_target_handle_t;
typedef void* ultimate_font_handle_t;
typedef void* ultimate_mesh_handle_t;

/* Graphics system initialization */
ultimate_error_t ultimate_graphics_init(const ultimate_graphics_config_t* config);
ultimate_error_t ultimate_graphics_shutdown(void);
ultimate_error_t ultimate_graphics_create_context(ultimate_window_handle_t window, 
                                                 ultimate_graphics_context_type_t type,
                                                 ultimate_graphics_context_t* context);
ultimate_error_t ultimate_graphics_destroy_context(ultimate_graphics_context_t context);
ultimate_error_t ultimate_graphics_make_current(ultimate_graphics_context_t context);
ultimate_error_t ultimate_graphics_swap_buffers(ultimate_graphics_context_t context);

/* Window management */
ultimate_error_t ultimate_window_create_ex(const ultimate_window_config_t* config,
                                          ultimate_window_style_t style,
                                          ultimate_window_handle_t parent,
                                          ultimate_window_handle_t* window);
ultimate_error_t ultimate_window_destroy(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_show(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_hide(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_minimize(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_maximize(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_restore(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_set_position(ultimate_window_handle_t window, int32_t x, int32_t y);
ultimate_error_t ultimate_window_get_position(ultimate_window_handle_t window, int32_t* x, int32_t* y);
ultimate_error_t ultimate_window_set_size(ultimate_window_handle_t window, uint32_t width, uint32_t height);
ultimate_error_t ultimate_window_get_size(ultimate_window_handle_t window, uint32_t* width, uint32_t* height);
ultimate_error_t ultimate_window_set_title(ultimate_window_handle_t window, const char* title);
ultimate_error_t ultimate_window_get_title(ultimate_window_handle_t window, char* buffer, uint32_t buffer_size);
ultimate_error_t ultimate_window_set_icon(ultimate_window_handle_t window, ultimate_texture_handle_t icon);
ultimate_error_t ultimate_window_center(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_set_topmost(ultimate_window_handle_t window, bool topmost);
ultimate_error_t ultimate_window_set_opacity(ultimate_window_handle_t window, float opacity);
ultimate_error_t ultimate_window_get_opacity(ultimate_window_handle_t window, float* opacity);
ultimate_error_t ultimate_window_set_fullscreen(ultimate_window_handle_t window, bool fullscreen);
ultimate_error_t ultimate_window_is_fullscreen(ultimate_window_handle_t window, bool* fullscreen);
ultimate_error_t ultimate_window_capture_mouse(ultimate_window_handle_t window);
ultimate_error_t ultimate_window_release_mouse(ultimate_window_handle_t window);

/* Texture management */
ultimate_error_t ultimate_texture_create(const ultimate_texture_desc_t* desc, ultimate_texture_handle_t* texture);
ultimate_error_t ultimate_texture_destroy(ultimate_texture_handle_t texture);
ultimate_error_t ultimate_texture_load_from_file(const char* filename, ultimate_texture_handle_t* texture);
ultimate_error_t ultimate_texture_load_from_memory(const void* data, uint32_t size, ultimate_texture_handle_t* texture);
ultimate_error_t ultimate_texture_save_to_file(ultimate_texture_handle_t texture, const char* filename);
ultimate_error_t ultimate_texture_update(ultimate_texture_handle_t texture, const void* data, uint32_t size);
ultimate_error_t ultimate_texture_get_info(ultimate_texture_handle_t texture, ultimate_texture_desc_t* desc);
ultimate_error_t ultimate_texture_generate_mipmaps(ultimate_texture_handle_t texture);
ultimate_error_t ultimate_texture_set_filter(ultimate_texture_handle_t texture, bool linear);
ultimate_error_t ultimate_texture_set_wrap(ultimate_texture_handle_t texture, bool repeat);
ultimate_error_t ultimate_texture_bind(ultimate_texture_handle_t texture, uint32_t slot);
ultimate_error_t ultimate_texture_unbind(uint32_t slot);

/* Shader management */
ultimate_error_t ultimate_shader_create_from_source(const char* vertex_source, const char* fragment_source, ultimate_shader_handle_t* shader);
ultimate_error_t ultimate_shader_create_from_file(const char* vertex_file, const char* fragment_file, ultimate_shader_handle_t* shader);
ultimate_error_t ultimate_shader_destroy(ultimate_shader_handle_t shader);
ultimate_error_t ultimate_shader_bind(ultimate_shader_handle_t shader);
ultimate_error_t ultimate_shader_unbind(void);
ultimate_error_t ultimate_shader_set_uniform_int(ultimate_shader_handle_t shader, const char* name, int32_t value);
ultimate_error_t ultimate_shader_set_uniform_float(ultimate_shader_handle_t shader, const char* name, float value);
ultimate_error_t ultimate_shader_set_uniform_vector2(ultimate_shader_handle_t shader, const char* name, const ultimate_vector2_t* value);
ultimate_error_t ultimate_shader_set_uniform_vector3(ultimate_shader_handle_t shader, const char* name, const ultimate_vector3_t* value);
ultimate_error_t ultimate_shader_set_uniform_vector4(ultimate_shader_handle_t shader, const char* name, const ultimate_vector4_t* value);
ultimate_error_t ultimate_shader_set_uniform_matrix4(ultimate_shader_handle_t shader, const char* name, const float* matrix);
ultimate_error_t ultimate_shader_set_uniform_texture(ultimate_shader_handle_t shader, const char* name, ultimate_texture_handle_t texture, uint32_t slot);

/* Buffer management */
ultimate_error_t ultimate_buffer_create_vertex(const void* data, uint32_t size, ultimate_buffer_handle_t* buffer);
ultimate_error_t ultimate_buffer_create_index(const void* data, uint32_t size, ultimate_buffer_handle_t* buffer);
ultimate_error_t ultimate_buffer_create_uniform(const void* data, uint32_t size, ultimate_buffer_handle_t* buffer);
ultimate_error_t ultimate_buffer_destroy(ultimate_buffer_handle_t buffer);
ultimate_error_t ultimate_buffer_update(ultimate_buffer_handle_t buffer, const void* data, uint32_t size, uint32_t offset);
ultimate_error_t ultimate_buffer_bind_vertex(ultimate_buffer_handle_t buffer);
ultimate_error_t ultimate_buffer_bind_index(ultimate_buffer_handle_t buffer);
ultimate_error_t ultimate_buffer_bind_uniform(ultimate_buffer_handle_t buffer, uint32_t slot);
ultimate_error_t ultimate_buffer_unbind_vertex(void);
ultimate_error_t ultimate_buffer_unbind_index(void);
ultimate_error_t ultimate_buffer_unbind_uniform(uint32_t slot);

/* Render target management */
ultimate_error_t ultimate_render_target_create(uint32_t width, uint32_t height, ultimate_texture_format_t format, ultimate_render_target_handle_t* target);
ultimate_error_t ultimate_render_target_destroy(ultimate_render_target_handle_t target);
ultimate_error_t ultimate_render_target_bind(ultimate_render_target_handle_t target);
ultimate_error_t ultimate_render_target_unbind(void);
ultimate_error_t ultimate_render_target_get_texture(ultimate_render_target_handle_t target, ultimate_texture_handle_t* texture);
ultimate_error_t ultimate_render_target_clear(ultimate_render_target_handle_t target, const ultimate_color_t* color);
ultimate_error_t ultimate_render_target_resize(ultimate_render_target_handle_t target, uint32_t width, uint32_t height);

/* Rendering functions */
ultimate_error_t ultimate_render_clear(const ultimate_color_t* color);
ultimate_error_t ultimate_render_clear_depth(float depth);
ultimate_error_t ultimate_render_set_viewport(const ultimate_rect_t* viewport);
ultimate_error_t ultimate_render_get_viewport(ultimate_rect_t* viewport);
ultimate_error_t ultimate_render_set_scissor(const ultimate_rect_t* scissor);
ultimate_error_t ultimate_render_disable_scissor(void);
ultimate_error_t ultimate_render_set_blend_mode(ultimate_blend_mode_t mode);
ultimate_error_t ultimate_render_set_depth_test(bool enable);
ultimate_error_t ultimate_render_set_depth_write(bool enable);
ultimate_error_t ultimate_render_set_cull_mode(bool enable, bool front_face);
ultimate_error_t ultimate_render_set_wireframe(bool enable);

/* Drawing functions */
ultimate_error_t ultimate_draw_arrays(ultimate_primitive_type_t primitive, uint32_t first, uint32_t count);
ultimate_error_t ultimate_draw_elements(ultimate_primitive_type_t primitive, uint32_t count, const uint32_t* indices);
ultimate_error_t ultimate_draw_instanced(ultimate_primitive_type_t primitive, uint32_t count, uint32_t instances);
ultimate_error_t ultimate_draw_point(const ultimate_vector2_t* position, const ultimate_color_t* color);
ultimate_error_t ultimate_draw_line(const ultimate_vector2_t* start, const ultimate_vector2_t* end, const ultimate_color_t* color);
ultimate_error_t ultimate_draw_rectangle(const ultimate_rect_t* rect, const ultimate_color_t* color);
ultimate_error_t ultimate_draw_rectangle_filled(const ultimate_rect_t* rect, const ultimate_color_t* color);
ultimate_error_t ultimate_draw_circle(const ultimate_vector2_t* center, float radius, const ultimate_color_t* color);
ultimate_error_t ultimate_draw_circle_filled(const ultimate_vector2_t* center, float radius, const ultimate_color_t* color);
ultimate_error_t ultimate_draw_triangle(const ultimate_vector2_t* p1, const ultimate_vector2_t* p2, const ultimate_vector2_t* p3, const ultimate_color_t* color);
ultimate_error_t ultimate_draw_triangle_filled(const ultimate_vector2_t* p1, const ultimate_vector2_t* p2, const ultimate_vector2_t* p3, const ultimate_color_t* color);
ultimate_error_t ultimate_draw_quad(const ultimate_vector2_t* position, const ultimate_vector2_t* size, ultimate_texture_handle_t texture);
ultimate_error_t ultimate_draw_sprite(ultimate_texture_handle_t texture, const ultimate_vector2_t* position, const ultimate_vector2_t* scale, float rotation);

/* Font and text rendering */
ultimate_error_t ultimate_font_load_from_file(const char* filename, uint32_t size, ultimate_font_handle_t* font);
ultimate_error_t ultimate_font_load_from_memory(const void* data, uint32_t size, uint32_t font_size, ultimate_font_handle_t* font);
ultimate_error_t ultimate_font_destroy(ultimate_font_handle_t font);
ultimate_error_t ultimate_font_get_text_size(ultimate_font_handle_t font, const char* text, ultimate_vector2_t* size);
ultimate_error_t ultimate_font_render_text(ultimate_font_handle_t font, const char* text, const ultimate_vector2_t* position, const ultimate_color_t* color);
ultimate_error_t ultimate_font_render_text_formatted(ultimate_font_handle_t font, const ultimate_vector2_t* position, const ultimate_color_t* color, const char* format, ...);

/* Mesh management */
ultimate_error_t ultimate_mesh_create(const ultimate_vertex_t* vertices, uint32_t vertex_count, const uint32_t* indices, uint32_t index_count, ultimate_mesh_handle_t* mesh);
ultimate_error_t ultimate_mesh_destroy(ultimate_mesh_handle_t mesh);
ultimate_error_t ultimate_mesh_load_from_file(const char* filename, ultimate_mesh_handle_t* mesh);
ultimate_error_t ultimate_mesh_save_to_file(ultimate_mesh_handle_t mesh, const char* filename);
ultimate_error_t ultimate_mesh_render(ultimate_mesh_handle_t mesh);
ultimate_error_t ultimate_mesh_get_bounds(ultimate_mesh_handle_t mesh, ultimate_vector3_t* min_bounds, ultimate_vector3_t* max_bounds);
ultimate_error_t ultimate_mesh_transform(ultimate_mesh_handle_t mesh, const float* transform_matrix);

/* Color utilities */
ultimate_error_t ultimate_color_create_rgb(uint8_t r, uint8_t g, uint8_t b, ultimate_color_t* color);
ultimate_error_t ultimate_color_create_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a, ultimate_color_t* color);
ultimate_error_t ultimate_color_create_hsv(float h, float s, float v, ultimate_color_t* color);
ultimate_error_t ultimate_color_create_hsl(float h, float s, float l, ultimate_color_t* color);
ultimate_error_t ultimate_color_to_rgb(const ultimate_color_t* color, uint8_t* r, uint8_t* g, uint8_t* b);
ultimate_error_t ultimate_color_to_rgba(const ultimate_color_t* color, uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a);
ultimate_error_t ultimate_color_to_hsv(const ultimate_color_t* color, float* h, float* s, float* v);
ultimate_error_t ultimate_color_to_hsl(const ultimate_color_t* color, float* h, float* s, float* l);
ultimate_error_t ultimate_color_lerp(const ultimate_color_t* a, const ultimate_color_t* b, float t, ultimate_color_t* result);
ultimate_error_t ultimate_color_multiply(const ultimate_color_t* a, const ultimate_color_t* b, ultimate_color_t* result);

/* Math utilities */
ultimate_error_t ultimate_vector2_create(float x, float y, ultimate_vector2_t* vector);
ultimate_error_t ultimate_vector2_add(const ultimate_vector2_t* a, const ultimate_vector2_t* b, ultimate_vector2_t* result);
ultimate_error_t ultimate_vector2_subtract(const ultimate_vector2_t* a, const ultimate_vector2_t* b, ultimate_vector2_t* result);
ultimate_error_t ultimate_vector2_multiply(const ultimate_vector2_t* a, float scalar, ultimate_vector2_t* result);
ultimate_error_t ultimate_vector2_dot(const ultimate_vector2_t* a, const ultimate_vector2_t* b, float* result);
ultimate_error_t ultimate_vector2_length(const ultimate_vector2_t* vector, float* length);
ultimate_error_t ultimate_vector2_normalize(const ultimate_vector2_t* vector, ultimate_vector2_t* result);
ultimate_error_t ultimate_vector2_distance(const ultimate_vector2_t* a, const ultimate_vector2_t* b, float* distance);

ultimate_error_t ultimate_vector3_create(float x, float y, float z, ultimate_vector3_t* vector);
ultimate_error_t ultimate_vector3_add(const ultimate_vector3_t* a, const ultimate_vector3_t* b, ultimate_vector3_t* result);
ultimate_error_t ultimate_vector3_subtract(const ultimate_vector3_t* a, const ultimate_vector3_t* b, ultimate_vector3_t* result);
ultimate_error_t ultimate_vector3_multiply(const ultimate_vector3_t* a, float scalar, ultimate_vector3_t* result);
ultimate_error_t ultimate_vector3_dot(const ultimate_vector3_t* a, const ultimate_vector3_t* b, float* result);
ultimate_error_t ultimate_vector3_cross(const ultimate_vector3_t* a, const ultimate_vector3_t* b, ultimate_vector3_t* result);
ultimate_error_t ultimate_vector3_length(const ultimate_vector3_t* vector, float* length);
ultimate_error_t ultimate_vector3_normalize(const ultimate_vector3_t* vector, ultimate_vector3_t* result);
ultimate_error_t ultimate_vector3_distance(const ultimate_vector3_t* a, const ultimate_vector3_t* b, float* distance);

/* Matrix utilities */
ultimate_error_t ultimate_matrix4_identity(float* matrix);
ultimate_error_t ultimate_matrix4_translate(float* matrix, const ultimate_vector3_t* translation);
ultimate_error_t ultimate_matrix4_rotate(float* matrix, const ultimate_vector3_t* axis, float angle);
ultimate_error_t ultimate_matrix4_scale(float* matrix, const ultimate_vector3_t* scale);
ultimate_error_t ultimate_matrix4_multiply(const float* a, const float* b, float* result);
ultimate_error_t ultimate_matrix4_inverse(const float* matrix, float* result);
ultimate_error_t ultimate_matrix4_transpose(const float* matrix, float* result);
ultimate_error_t ultimate_matrix4_perspective(float fov, float aspect, float near_plane, float far_plane, float* matrix);
ultimate_error_t ultimate_matrix4_orthographic(float left, float right, float bottom, float top, float near_plane, float far_plane, float* matrix);
ultimate_error_t ultimate_matrix4_look_at(const ultimate_vector3_t* eye, const ultimate_vector3_t* center, const ultimate_vector3_t* up, float* matrix);

/* Input handling */
ultimate_error_t ultimate_input_register_callback(ultimate_window_handle_t window, ultimate_input_callback_t callback, void* user_data);
ultimate_error_t ultimate_input_unregister_callback(ultimate_window_handle_t window);
ultimate_error_t ultimate_input_get_mouse_position(ultimate_window_handle_t window, int32_t* x, int32_t* y);
ultimate_error_t ultimate_input_set_mouse_position(ultimate_window_handle_t window, int32_t x, int32_t y);
ultimate_error_t ultimate_input_is_key_pressed(uint32_t key_code, bool* pressed);
ultimate_error_t ultimate_input_is_mouse_button_pressed(uint32_t button, bool* pressed);
ultimate_error_t ultimate_input_get_mouse_wheel_delta(float* delta);

/* Performance and debugging */
ultimate_error_t ultimate_graphics_get_stats(ultimate_system_stats_t* stats);
ultimate_error_t ultimate_graphics_begin_frame(void);
ultimate_error_t ultimate_graphics_end_frame(void);
ultimate_error_t ultimate_graphics_present(void);
ultimate_error_t ultimate_graphics_wait_for_vsync(void);
ultimate_error_t ultimate_graphics_screenshot(const char* filename);
ultimate_error_t ultimate_graphics_enable_debug_layer(bool enable);
ultimate_error_t ultimate_graphics_set_debug_callback(void (*callback)(const char* message));

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_GRAPHICS_H */