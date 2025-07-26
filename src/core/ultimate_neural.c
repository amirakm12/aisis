#include "ultimate_neural.h"
#include "ultimate_core.h"
#include "ultimate_memory.h"
#include "ultimate_errors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Neural model structure */
typedef struct {
    uint32_t id;
    ultimate_neural_config_t config;
    uint32_t layer_count;
    uint32_t* layer_sizes;
    float** weights;
    float** biases;
    float** activations;
    bool is_trained;
    uint32_t epoch_count;
    float learning_rate;
    float loss;
} ultimate_neural_model_impl_t;

/* Global neural system state */
static ultimate_neural_model_impl_t g_models[ULTIMATE_MAX_NEURAL_MODELS];
static uint32_t g_model_count = 0;
static uint32_t g_next_model_id = 1;
static bool g_neural_system_initialized = false;

/* Activation functions */
static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static float tanh_activation(float x) {
    return tanhf(x);
}

static float leaky_relu(float x) {
    return x > 0.0f ? x : 0.01f * x;
}

static float softmax_helper(const float* input, uint32_t size, uint32_t index) {
    float max_val = input[0];
    for (uint32_t i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        sum += expf(input[i] - max_val);
    }
    
    return expf(input[index] - max_val) / sum;
}

/* Neural system initialization */
ultimate_error_t ultimate_neural_init(void) {
    if (g_neural_system_initialized) {
        return ULTIMATE_ERROR_ALREADY_INITIALIZED;
    }
    
    memset(g_models, 0, sizeof(g_models));
    g_model_count = 0;
    g_next_model_id = 1;
    
    g_neural_system_initialized = true;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_shutdown(void) {
    if (!g_neural_system_initialized) {
        return ULTIMATE_ERROR_NOT_INITIALIZED;
    }
    
    /* Cleanup all models */
    for (uint32_t i = 0; i < g_model_count; i++) {
        ultimate_neural_model_destroy((ultimate_neural_model_t)&g_models[i]);
    }
    
    g_neural_system_initialized = false;
    return ULTIMATE_OK;
}

/* Model management */
ultimate_error_t ultimate_neural_model_create(const ultimate_neural_config_t* config,
                                             ultimate_neural_model_t* model) {
    if (!config || !model || g_model_count >= ULTIMATE_MAX_NEURAL_MODELS) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    if (!g_neural_system_initialized) {
        ultimate_neural_init();
    }
    
    ultimate_neural_model_impl_t* impl = &g_models[g_model_count];
    memset(impl, 0, sizeof(ultimate_neural_model_impl_t));
    
    impl->id = g_next_model_id++;
    impl->config = *config;
    impl->learning_rate = 0.001f;  /* Default learning rate */
    
    *model = (ultimate_neural_model_t)impl;
    g_model_count++;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_model_destroy(ultimate_neural_model_t model) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    if (!impl) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    /* Free allocated memory */
    if (impl->layer_sizes) {
        ultimate_free(impl->layer_sizes);
    }
    
    if (impl->weights) {
        for (uint32_t i = 0; i < impl->layer_count - 1; i++) {
            if (impl->weights[i]) {
                ultimate_free(impl->weights[i]);
            }
        }
        ultimate_free(impl->weights);
    }
    
    if (impl->biases) {
        for (uint32_t i = 0; i < impl->layer_count - 1; i++) {
            if (impl->biases[i]) {
                ultimate_free(impl->biases[i]);
            }
        }
        ultimate_free(impl->biases);
    }
    
    if (impl->activations) {
        for (uint32_t i = 0; i < impl->layer_count; i++) {
            if (impl->activations[i]) {
                ultimate_free(impl->activations[i]);
            }
        }
        ultimate_free(impl->activations);
    }
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_model_add_layer(ultimate_neural_model_t model, uint32_t size) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    if (!impl) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    /* Reallocate layer sizes array */
    uint32_t* new_sizes = (uint32_t*)ultimate_realloc(impl->layer_sizes, 
                                                     (impl->layer_count + 1) * sizeof(uint32_t));
    if (!new_sizes) {
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    impl->layer_sizes = new_sizes;
    impl->layer_sizes[impl->layer_count] = size;
    impl->layer_count++;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_model_compile(ultimate_neural_model_t model) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    if (!impl || impl->layer_count < 2) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    /* Allocate weights */
    impl->weights = (float**)ultimate_malloc((impl->layer_count - 1) * sizeof(float*));
    if (!impl->weights) {
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Allocate biases */
    impl->biases = (float**)ultimate_malloc((impl->layer_count - 1) * sizeof(float*));
    if (!impl->biases) {
        ultimate_free(impl->weights);
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Allocate activations */
    impl->activations = (float**)ultimate_malloc(impl->layer_count * sizeof(float*));
    if (!impl->activations) {
        ultimate_free(impl->weights);
        ultimate_free(impl->biases);
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Initialize weights and biases */
    for (uint32_t i = 0; i < impl->layer_count - 1; i++) {
        uint32_t input_size = impl->layer_sizes[i];
        uint32_t output_size = impl->layer_sizes[i + 1];
        
        /* Allocate weights matrix */
        impl->weights[i] = (float*)ultimate_malloc(input_size * output_size * sizeof(float));
        if (!impl->weights[i]) {
            return ULTIMATE_ERROR_OUT_OF_MEMORY;
        }
        
        /* Allocate bias vector */
        impl->biases[i] = (float*)ultimate_malloc(output_size * sizeof(float));
        if (!impl->biases[i]) {
            return ULTIMATE_ERROR_OUT_OF_MEMORY;
        }
        
        /* Initialize with Xavier/Glorot initialization */
        float scale = sqrtf(2.0f / (input_size + output_size));
        for (uint32_t j = 0; j < input_size * output_size; j++) {
            impl->weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
        
        /* Initialize biases to zero */
        memset(impl->biases[i], 0, output_size * sizeof(float));
    }
    
    /* Allocate activation arrays */
    for (uint32_t i = 0; i < impl->layer_count; i++) {
        impl->activations[i] = (float*)ultimate_malloc(impl->layer_sizes[i] * sizeof(float));
        if (!impl->activations[i]) {
            return ULTIMATE_ERROR_OUT_OF_MEMORY;
        }
    }
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_inference(ultimate_neural_model_t model,
                                          ultimate_neural_tensor_t input,
                                          ultimate_neural_tensor_t output) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    if (!impl || !input.data || !output.data) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    if (!impl->weights || !impl->biases || !impl->activations) {
        return ULTIMATE_ERROR_NOT_INITIALIZED;
    }
    
    /* Copy input to first activation layer */
    float* input_data = (float*)input.data;
    memcpy(impl->activations[0], input_data, impl->layer_sizes[0] * sizeof(float));
    
    /* Forward propagation */
    for (uint32_t layer = 0; layer < impl->layer_count - 1; layer++) {
        uint32_t input_size = impl->layer_sizes[layer];
        uint32_t output_size = impl->layer_sizes[layer + 1];
        
        /* Matrix multiplication: activations[layer+1] = weights[layer] * activations[layer] + biases[layer] */
        for (uint32_t j = 0; j < output_size; j++) {
            float sum = impl->biases[layer][j];
            for (uint32_t i = 0; i < input_size; i++) {
                sum += impl->weights[layer][i * output_size + j] * impl->activations[layer][i];
            }
            
            /* Apply activation function */
            switch (impl->config.model_type) {
                case ULTIMATE_NEURAL_TYPE_FEEDFORWARD:
                    impl->activations[layer + 1][j] = sigmoid(sum);
                    break;
                case ULTIMATE_NEURAL_TYPE_CNN:
                case ULTIMATE_NEURAL_TYPE_RNN:
                case ULTIMATE_NEURAL_TYPE_LSTM:
                    impl->activations[layer + 1][j] = tanh_activation(sum);
                    break;
                case ULTIMATE_NEURAL_TYPE_TRANSFORMER:
                    impl->activations[layer + 1][j] = relu(sum);
                    break;
                default:
                    impl->activations[layer + 1][j] = sum;
                    break;
            }
        }
    }
    
    /* Copy output */
    float* output_data = (float*)output.data;
    uint32_t output_layer = impl->layer_count - 1;
    memcpy(output_data, impl->activations[output_layer], 
           impl->layer_sizes[output_layer] * sizeof(float));
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_train_batch(ultimate_neural_model_t model,
                                            ultimate_neural_tensor_t* inputs,
                                            ultimate_neural_tensor_t* targets,
                                            uint32_t batch_size) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    if (!impl || !inputs || !targets || batch_size == 0) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    float total_loss = 0.0f;
    
    /* Process each sample in the batch */
    for (uint32_t sample = 0; sample < batch_size; sample++) {
        /* Forward pass */
        ultimate_error_t error = ultimate_neural_inference(model, inputs[sample], targets[sample]);
        if (error != ULTIMATE_OK) {
            return error;
        }
        
        /* Calculate loss (mean squared error) */
        float* target_data = (float*)targets[sample].data;
        uint32_t output_layer = impl->layer_count - 1;
        uint32_t output_size = impl->layer_sizes[output_layer];
        
        float sample_loss = 0.0f;
        for (uint32_t i = 0; i < output_size; i++) {
            float diff = impl->activations[output_layer][i] - target_data[i];
            sample_loss += diff * diff;
        }
        total_loss += sample_loss / output_size;
        
        /* Backward pass (simplified gradient descent) */
        /* This is a basic implementation - real neural networks would use more sophisticated algorithms */
        for (uint32_t layer = impl->layer_count - 2; layer < impl->layer_count; layer--) {
            uint32_t input_size = impl->layer_sizes[layer];
            uint32_t current_output_size = impl->layer_sizes[layer + 1];
            
            for (uint32_t j = 0; j < current_output_size; j++) {
                float error_term = (impl->activations[layer + 1][j] - target_data[j]) * 
                                  impl->activations[layer + 1][j] * 
                                  (1.0f - impl->activations[layer + 1][j]);
                
                /* Update weights */
                for (uint32_t i = 0; i < input_size; i++) {
                    impl->weights[layer][i * current_output_size + j] -= 
                        impl->learning_rate * error_term * impl->activations[layer][i];
                }
                
                /* Update bias */
                impl->biases[layer][j] -= impl->learning_rate * error_term;
            }
            
            if (layer == 0) break;  /* Prevent underflow */
        }
    }
    
    impl->loss = total_loss / batch_size;
    impl->epoch_count++;
    impl->is_trained = true;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_model_save(ultimate_neural_model_t model, const char* filename) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    if (!impl || !filename) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        return ULTIMATE_ERROR_FILE_NOT_FOUND;
    }
    
    /* Write header */
    fwrite(&impl->config, sizeof(ultimate_neural_config_t), 1, file);
    fwrite(&impl->layer_count, sizeof(uint32_t), 1, file);
    fwrite(&impl->learning_rate, sizeof(float), 1, file);
    fwrite(&impl->epoch_count, sizeof(uint32_t), 1, file);
    
    /* Write layer sizes */
    fwrite(impl->layer_sizes, sizeof(uint32_t), impl->layer_count, file);
    
    /* Write weights and biases */
    for (uint32_t i = 0; i < impl->layer_count - 1; i++) {
        uint32_t input_size = impl->layer_sizes[i];
        uint32_t output_size = impl->layer_sizes[i + 1];
        
        fwrite(impl->weights[i], sizeof(float), input_size * output_size, file);
        fwrite(impl->biases[i], sizeof(float), output_size, file);
    }
    
    fclose(file);
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_model_load(const char* filename, ultimate_neural_model_t* model) {
    if (!filename || !model) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return ULTIMATE_ERROR_FILE_NOT_FOUND;
    }
    
    /* Create new model */
    ultimate_neural_config_t config;
    if (fread(&config, sizeof(ultimate_neural_config_t), 1, file) != 1) {
        fclose(file);
        return ULTIMATE_ERROR_IO_ERROR;
    }
    
    ultimate_error_t error = ultimate_neural_model_create(&config, model);
    if (error != ULTIMATE_OK) {
        fclose(file);
        return error;
    }
    
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)*model;
    
    /* Read header */
    if (fread(&impl->layer_count, sizeof(uint32_t), 1, file) != 1 ||
        fread(&impl->learning_rate, sizeof(float), 1, file) != 1 ||
        fread(&impl->epoch_count, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        ultimate_neural_model_destroy(*model);
        return ULTIMATE_ERROR_IO_ERROR;
    }
    
    /* Allocate and read layer sizes */
    impl->layer_sizes = (uint32_t*)ultimate_malloc(impl->layer_count * sizeof(uint32_t));
    if (!impl->layer_sizes || 
        fread(impl->layer_sizes, sizeof(uint32_t), impl->layer_count, file) != impl->layer_count) {
        fclose(file);
        ultimate_neural_model_destroy(*model);
        return ULTIMATE_ERROR_IO_ERROR;
    }
    
    /* Compile model to allocate arrays */
    error = ultimate_neural_model_compile(*model);
    if (error != ULTIMATE_OK) {
        fclose(file);
        ultimate_neural_model_destroy(*model);
        return error;
    }
    
    /* Read weights and biases */
    for (uint32_t i = 0; i < impl->layer_count - 1; i++) {
        uint32_t input_size = impl->layer_sizes[i];
        uint32_t output_size = impl->layer_sizes[i + 1];
        
        if (fread(impl->weights[i], sizeof(float), input_size * output_size, file) != input_size * output_size ||
            fread(impl->biases[i], sizeof(float), output_size, file) != output_size) {
            fclose(file);
            ultimate_neural_model_destroy(*model);
            return ULTIMATE_ERROR_IO_ERROR;
        }
    }
    
    impl->is_trained = true;
    fclose(file);
    return ULTIMATE_OK;
}

/* Tensor operations */
ultimate_error_t ultimate_neural_tensor_create(uint32_t* shape, uint32_t ndim,
                                              ultimate_neural_precision_t precision,
                                              ultimate_neural_tensor_t* tensor) {
    if (!shape || !tensor || ndim == 0) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    /* Calculate total size */
    uint32_t total_size = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    
    /* Calculate element size based on precision */
    uint32_t element_size;
    switch (precision) {
        case ULTIMATE_NEURAL_PRECISION_FLOAT32:
            element_size = sizeof(float);
            break;
        case ULTIMATE_NEURAL_PRECISION_FLOAT16:
            element_size = 2;
            break;
        case ULTIMATE_NEURAL_PRECISION_INT8:
            element_size = sizeof(int8_t);
            break;
        case ULTIMATE_NEURAL_PRECISION_INT16:
            element_size = sizeof(int16_t);
            break;
        default:
            return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    /* Allocate tensor */
    tensor->data = ultimate_malloc(total_size * element_size);
    if (!tensor->data) {
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    tensor->shape = (uint32_t*)ultimate_malloc(ndim * sizeof(uint32_t));
    if (!tensor->shape) {
        ultimate_free(tensor->data);
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy(tensor->shape, shape, ndim * sizeof(uint32_t));
    tensor->ndim = ndim;
    tensor->precision = precision;
    tensor->size = total_size * element_size;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_tensor_destroy(ultimate_neural_tensor_t* tensor) {
    if (!tensor) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    if (tensor->data) {
        ultimate_free(tensor->data);
        tensor->data = NULL;
    }
    
    if (tensor->shape) {
        ultimate_free(tensor->shape);
        tensor->shape = NULL;
    }
    
    tensor->ndim = 0;
    tensor->size = 0;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_tensor_fill(ultimate_neural_tensor_t* tensor, float value) {
    if (!tensor || !tensor->data) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    uint32_t total_elements = tensor->size;
    switch (tensor->precision) {
        case ULTIMATE_NEURAL_PRECISION_FLOAT32:
            total_elements /= sizeof(float);
            break;
        case ULTIMATE_NEURAL_PRECISION_FLOAT16:
            total_elements /= 2;
            break;
        case ULTIMATE_NEURAL_PRECISION_INT8:
            total_elements /= sizeof(int8_t);
            break;
        case ULTIMATE_NEURAL_PRECISION_INT16:
            total_elements /= sizeof(int16_t);
            break;
    }
    
    if (tensor->precision == ULTIMATE_NEURAL_PRECISION_FLOAT32) {
        float* data = (float*)tensor->data;
        for (uint32_t i = 0; i < total_elements; i++) {
            data[i] = value;
        }
    } else {
        /* For other precisions, convert and fill */
        for (uint32_t i = 0; i < total_elements; i++) {
            switch (tensor->precision) {
                case ULTIMATE_NEURAL_PRECISION_INT8:
                    ((int8_t*)tensor->data)[i] = (int8_t)value;
                    break;
                case ULTIMATE_NEURAL_PRECISION_INT16:
                    ((int16_t*)tensor->data)[i] = (int16_t)value;
                    break;
                default:
                    break;
            }
        }
    }
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_neural_tensor_copy(const ultimate_neural_tensor_t* src,
                                            ultimate_neural_tensor_t* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    if (src->size != dst->size || src->precision != dst->precision) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    memcpy(dst->data, src->data, src->size);
    return ULTIMATE_OK;
}

/* Model information */
uint32_t ultimate_neural_model_get_layer_count(ultimate_neural_model_t model) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    return impl ? impl->layer_count : 0;
}

uint32_t ultimate_neural_model_get_parameter_count(ultimate_neural_model_t model) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    if (!impl || !impl->layer_sizes) {
        return 0;
    }
    
    uint32_t param_count = 0;
    for (uint32_t i = 0; i < impl->layer_count - 1; i++) {
        uint32_t input_size = impl->layer_sizes[i];
        uint32_t output_size = impl->layer_sizes[i + 1];
        param_count += input_size * output_size + output_size;  /* weights + biases */
    }
    
    return param_count;
}

float ultimate_neural_model_get_loss(ultimate_neural_model_t model) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    return impl ? impl->loss : 0.0f;
}

uint32_t ultimate_neural_model_get_epoch_count(ultimate_neural_model_t model) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    return impl ? impl->epoch_count : 0;
}

bool ultimate_neural_model_is_trained(ultimate_neural_model_t model) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    return impl ? impl->is_trained : false;
}

ultimate_error_t ultimate_neural_model_set_learning_rate(ultimate_neural_model_t model, float rate) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    if (!impl || rate <= 0.0f) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    impl->learning_rate = rate;
    return ULTIMATE_OK;
}

float ultimate_neural_model_get_learning_rate(ultimate_neural_model_t model) {
    ultimate_neural_model_impl_t* impl = (ultimate_neural_model_impl_t*)model;
    return impl ? impl->learning_rate : 0.0f;
}