#ifndef ULTIMATE_HARDWARE_H
#define ULTIMATE_HARDWARE_H

/**
 * @file ultimate_hardware.h
 * @brief ULTIMATE Hardware Abstraction Layer
 * @version 1.0.0
 * @date 2024
 * 
 * Hardware abstraction layer for the ULTIMATE embedded system.
 * Provides unified interface for GPIO, UART, SPI, I2C, and other peripherals.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "ultimate_types.h"
#include "ultimate_errors.h"

/* GPIO Management */
ultimate_error_t ultimate_gpio_init(uint32_t port, uint32_t pin, ultimate_gpio_mode_t mode);
ultimate_error_t ultimate_gpio_deinit(uint32_t port, uint32_t pin);
ultimate_error_t ultimate_gpio_write(uint32_t port, uint32_t pin, ultimate_gpio_state_t state);
ultimate_gpio_state_t ultimate_gpio_read(uint32_t port, uint32_t pin);
ultimate_error_t ultimate_gpio_toggle(uint32_t port, uint32_t pin);

/* UART Management */
typedef struct ultimate_uart* ultimate_uart_handle_t;

typedef struct {
    uint32_t baudrate;
    uint32_t data_bits;
    uint32_t stop_bits;
    uint32_t parity;
    bool flow_control;
} ultimate_uart_config_t;

ultimate_error_t ultimate_uart_init(uint32_t instance, 
                                   const ultimate_uart_config_t* config,
                                   ultimate_uart_handle_t* uart);
ultimate_error_t ultimate_uart_deinit(ultimate_uart_handle_t uart);
ultimate_error_t ultimate_uart_send(ultimate_uart_handle_t uart, 
                                   const uint8_t* data, 
                                   size_t size,
                                   ultimate_timeout_t timeout);
ultimate_error_t ultimate_uart_receive(ultimate_uart_handle_t uart,
                                      uint8_t* data,
                                      size_t max_size,
                                      size_t* received_size,
                                      ultimate_timeout_t timeout);

/* SPI Management */
typedef struct ultimate_spi* ultimate_spi_handle_t;

typedef struct {
    uint32_t frequency;
    uint32_t mode;
    uint32_t data_size;
    bool msb_first;
} ultimate_spi_config_t;

ultimate_error_t ultimate_spi_init(uint32_t instance,
                                  const ultimate_spi_config_t* config,
                                  ultimate_spi_handle_t* spi);
ultimate_error_t ultimate_spi_deinit(ultimate_spi_handle_t spi);
ultimate_error_t ultimate_spi_transfer(ultimate_spi_handle_t spi,
                                      const uint8_t* tx_data,
                                      uint8_t* rx_data,
                                      size_t size,
                                      ultimate_timeout_t timeout);

/* I2C Management */
typedef struct ultimate_i2c* ultimate_i2c_handle_t;

typedef struct {
    uint32_t frequency;
    uint32_t address_mode;
    bool general_call;
} ultimate_i2c_config_t;

ultimate_error_t ultimate_i2c_init(uint32_t instance,
                                  const ultimate_i2c_config_t* config,
                                  ultimate_i2c_handle_t* i2c);
ultimate_error_t ultimate_i2c_deinit(ultimate_i2c_handle_t i2c);
ultimate_error_t ultimate_i2c_write(ultimate_i2c_handle_t i2c,
                                   uint16_t device_addr,
                                   const uint8_t* data,
                                   size_t size,
                                   ultimate_timeout_t timeout);
ultimate_error_t ultimate_i2c_read(ultimate_i2c_handle_t i2c,
                                  uint16_t device_addr,
                                  uint8_t* data,
                                  size_t size,
                                  ultimate_timeout_t timeout);

/* ADC Management */
typedef struct ultimate_adc* ultimate_adc_handle_t;

ultimate_error_t ultimate_adc_init(uint32_t instance, ultimate_adc_handle_t* adc);
ultimate_error_t ultimate_adc_deinit(ultimate_adc_handle_t adc);
ultimate_error_t ultimate_adc_read(ultimate_adc_handle_t adc, 
                                  uint32_t channel, 
                                  uint16_t* value);

/* PWM Management */
typedef struct ultimate_pwm* ultimate_pwm_handle_t;

typedef struct {
    uint32_t frequency;
    uint32_t duty_cycle_percent;
    bool invert_output;
} ultimate_pwm_config_t;

ultimate_error_t ultimate_pwm_init(uint32_t instance,
                                  const ultimate_pwm_config_t* config,
                                  ultimate_pwm_handle_t* pwm);
ultimate_error_t ultimate_pwm_deinit(ultimate_pwm_handle_t pwm);
ultimate_error_t ultimate_pwm_start(ultimate_pwm_handle_t pwm);
ultimate_error_t ultimate_pwm_stop(ultimate_pwm_handle_t pwm);
ultimate_error_t ultimate_pwm_set_duty_cycle(ultimate_pwm_handle_t pwm, 
                                            uint32_t duty_cycle_percent);

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_HARDWARE_H */