#ifndef ULTIMATE_NEURAL_H
#define ULTIMATE_NEURAL_H

/**
 * @file ultimate_neural.h
 * @brief ULTIMATE Neural Processing System
 * @version 1.0.0
 * @date 2024
 * 
 * Neural network processing and AI functionality for the ULTIMATE Windows system.
 * Provides inference, training, and model management capabilities.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "ultimate_types.h"
#include "ultimate_errors.h"

/* Neural network types */
typedef enum {
    ULTIMATE_NEURAL_TYPE_FEEDFORWARD = 0,
    ULTIMATE_NEURAL_TYPE_CNN,
    ULTIMATE_NEURAL_TYPE_RNN,
    ULTIMATE_NEURAL_TYPE_LSTM,
    ULTIMATE_NEURAL_TYPE_TRANSFORMER,
    ULTIMATE_NEURAL_TYPE_CUSTOM
} ultimate_neural_type_t;

typedef enum {
    ULTIMATE_ACTIVATION_LINEAR = 0,
    ULTIMATE_ACTIVATION_SIGMOID,
    ULTIMATE_ACTIVATION_TANH,
    ULTIMATE_ACTIVATION_RELU,
    ULTIMATE_ACTIVATION_LEAKY_RELU,
    ULTIMATE_ACTIVATION_SOFTMAX,
    ULTIMATE_ACTIVATION_SWISH
} ultimate_activation_t;

typedef enum {
    ULTIMATE_NEURAL_PRECISION_INT8 = 0,
    ULTIMATE_NEURAL_PRECISION_INT16,
    ULTIMATE_NEURAL_PRECISION_FLOAT16,
    ULTIMATE_NEURAL_PRECISION_FLOAT32
} ultimate_neural_precision_t;

/* Neural network handles */
typedef struct ultimate_neural_model* ultimate_neural_model_t;
typedef struct ultimate_neural_layer* ultimate_neural_layer_t;
typedef struct ultimate_neural_tensor* ultimate_neural_tensor_t;

/* Neural tensor structure */
typedef struct {
    void* data;
    uint32_t* dimensions;
    uint32_t num_dimensions;
    ultimate_neural_precision_t precision;
    size_t data_size;
    bool is_quantized;
} ultimate_tensor_info_t;

/* Neural layer configuration */
typedef struct {
    ultimate_neural_type_t type;
    uint32_t input_size;
    uint32_t output_size;
    ultimate_activation_t activation;
    float dropout_rate;
    bool use_bias;
    const char* name;
} ultimate_layer_config_t;

/* Neural model configuration */
typedef struct {
    ultimate_neural_type_t model_type;
    uint32_t num_layers;
    const ultimate_layer_config_t* layers;
    ultimate_neural_precision_t precision;
    size_t memory_limit;
    bool enable_quantization;
    bool enable_pruning;
    const char* name;
} ultimate_neural_config_t;

/* Training configuration */
typedef struct {
    float learning_rate;
    uint32_t batch_size;
    uint32_t epochs;
    float momentum;
    float weight_decay;
    bool enable_early_stopping;
    float validation_split;
    uint32_t patience;
} ultimate_training_config_t;

/* Neural processing statistics */
typedef struct {
    uint32_t inference_count;
    uint32_t training_steps;
    float average_inference_time_ms;
    float average_training_time_ms;
    size_t memory_usage;
    size_t peak_memory_usage;
    float model_accuracy;
    float validation_loss;
} ultimate_neural_stats_t;

/* Core neural API */
ultimate_error_t ultimate_neural_init(void);
ultimate_error_t ultimate_neural_deinit(void);

/* Model management */
ultimate_error_t ultimate_neural_model_create(const ultimate_neural_config_t* config,
                                             ultimate_neural_model_t* model);
ultimate_error_t ultimate_neural_model_delete(ultimate_neural_model_t model);
ultimate_error_t ultimate_neural_model_load(const char* model_path,
                                           ultimate_neural_model_t* model);
ultimate_error_t ultimate_neural_model_save(ultimate_neural_model_t model,
                                           const char* model_path);

/* Model information */
ultimate_error_t ultimate_neural_model_get_info(ultimate_neural_model_t model,
                                               ultimate_neural_config_t* info);
ultimate_error_t ultimate_neural_model_get_stats(ultimate_neural_model_t model,
                                                ultimate_neural_stats_t* stats);

/* Tensor operations */
ultimate_error_t ultimate_tensor_create(const uint32_t* dimensions,
                                       uint32_t num_dims,
                                       ultimate_neural_precision_t precision,
                                       ultimate_neural_tensor_t* tensor);
ultimate_error_t ultimate_tensor_delete(ultimate_neural_tensor_t tensor);
ultimate_error_t ultimate_tensor_get_info(ultimate_neural_tensor_t tensor,
                                         ultimate_tensor_info_t* info);

/* Data operations */
ultimate_error_t ultimate_tensor_set_data(ultimate_neural_tensor_t tensor,
                                         const void* data,
                                         size_t data_size);
ultimate_error_t ultimate_tensor_get_data(ultimate_neural_tensor_t tensor,
                                         void* data,
                                         size_t max_size,
                                         size_t* actual_size);

/* Inference operations */
ultimate_error_t ultimate_neural_inference(ultimate_neural_model_t model,
                                          ultimate_neural_tensor_t input,
                                          ultimate_neural_tensor_t output);

ultimate_error_t ultimate_neural_inference_batch(ultimate_neural_model_t model,
                                                ultimate_neural_tensor_t* inputs,
                                                ultimate_neural_tensor_t* outputs,
                                                uint32_t batch_size);

/* Training operations */
ultimate_error_t ultimate_neural_train(ultimate_neural_model_t model,
                                      ultimate_neural_tensor_t* training_data,
                                      ultimate_neural_tensor_t* labels,
                                      uint32_t num_samples,
                                      const ultimate_training_config_t* config);

ultimate_error_t ultimate_neural_validate(ultimate_neural_model_t model,
                                         ultimate_neural_tensor_t* validation_data,
                                         ultimate_neural_tensor_t* labels,
                                         uint32_t num_samples,
                                         float* accuracy);

/* Model optimization */
ultimate_error_t ultimate_neural_quantize_model(ultimate_neural_model_t model);
ultimate_error_t ultimate_neural_prune_model(ultimate_neural_model_t model,
                                            float pruning_ratio);
ultimate_error_t ultimate_neural_optimize_model(ultimate_neural_model_t model);

/* Hardware acceleration */
typedef enum {
    ULTIMATE_NEURAL_ACCEL_NONE = 0,
    ULTIMATE_NEURAL_ACCEL_CPU_SIMD,
    ULTIMATE_NEURAL_ACCEL_GPU,
    ULTIMATE_NEURAL_ACCEL_NPU,
    ULTIMATE_NEURAL_ACCEL_CUSTOM
} ultimate_neural_accelerator_t;

ultimate_error_t ultimate_neural_set_accelerator(ultimate_neural_accelerator_t accel);
ultimate_neural_accelerator_t ultimate_neural_get_accelerator(void);

/* Memory management for neural operations */
ultimate_error_t ultimate_neural_memory_pool_create(size_t pool_size);
ultimate_error_t ultimate_neural_memory_pool_destroy(void);
size_t ultimate_neural_get_memory_usage(void);

/* Model compilation and optimization */
ultimate_error_t ultimate_neural_compile_model(ultimate_neural_model_t model);
ultimate_error_t ultimate_neural_set_optimization_level(uint32_t level);

/* Callback functions */
typedef void (*ultimate_neural_progress_callback_t)(float progress, void* user_data);
typedef void (*ultimate_neural_error_callback_t)(ultimate_error_t error, 
                                                const char* message, 
                                                void* user_data);

ultimate_error_t ultimate_neural_set_progress_callback(ultimate_neural_progress_callback_t callback,
                                                      void* user_data);
ultimate_error_t ultimate_neural_set_error_callback(ultimate_neural_error_callback_t callback,
                                                   void* user_data);

/* Utility functions */
ultimate_error_t ultimate_neural_normalize_data(float* data, 
                                               uint32_t size,
                                               float min_val,
                                               float max_val);

ultimate_error_t ultimate_neural_denormalize_data(float* data,
                                                 uint32_t size,
                                                 float min_val,
                                                 float max_val);

/* Model serialization */
ultimate_error_t ultimate_neural_serialize_model(ultimate_neural_model_t model,
                                                void* buffer,
                                                size_t buffer_size,
                                                size_t* serialized_size);

ultimate_error_t ultimate_neural_deserialize_model(const void* buffer,
                                                  size_t buffer_size,
                                                  ultimate_neural_model_t* model);

/* Performance profiling */
typedef struct {
    float total_inference_time_ms;
    float layer_times_ms[32];  /* Max 32 layers */
    uint32_t num_layers;
    size_t memory_peak_bytes;
    uint32_t cache_hits;
    uint32_t cache_misses;
} ultimate_neural_profile_t;

ultimate_error_t ultimate_neural_enable_profiling(bool enable);
ultimate_error_t ultimate_neural_get_profile(ultimate_neural_model_t model,
                                            ultimate_neural_profile_t* profile);

/* Model verification and testing */
ultimate_error_t ultimate_neural_verify_model(ultimate_neural_model_t model);
ultimate_error_t ultimate_neural_test_model(ultimate_neural_model_t model,
                                           ultimate_neural_tensor_t* test_data,
                                           ultimate_neural_tensor_t* expected_outputs,
                                           uint32_t num_tests,
                                           float* accuracy);

/* Configuration and system settings */
typedef struct {
    size_t max_memory_usage;
    uint32_t max_models;
    uint32_t max_tensors;
    ultimate_neural_accelerator_t preferred_accelerator;
    bool enable_debug_mode;
    bool enable_profiling;
    uint32_t thread_count;
} ultimate_neural_system_config_t;

ultimate_error_t ultimate_neural_configure_system(const ultimate_neural_system_config_t* config);

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_NEURAL_H */ 