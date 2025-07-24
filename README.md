# AISIS Project

A comprehensive C/C++ embedded development project with ARM toolchain support.

## Overview

AISIS is a modular embedded software library designed for ARM-based systems. It provides device management, data processing, and communication capabilities with a clean, well-documented API.

## Features

- **Device Management**: Scan, connect, and manage multiple devices
- **Data Processing**: Efficient data transformation and processing
- **Command Interface**: Send commands and receive responses from devices
- **Cross-Platform**: Supports both native and ARM embedded targets
- **Modular Design**: Clean separation between core library and application
- **Comprehensive Testing**: Unit tests and integration tests included

## Project Structure

```
├── src/                    # Main application source files
│   └── main.c             # Main application entry point
├── include/               # Main application headers
│   └── aisis.h           # Main AISIS API header
├── aisis/                 # AISIS library
│   ├── src/              # Library source files
│   │   ├── aisis_core.c  # Core library functions
│   │   ├── aisis_device.c # Device management
│   │   └── aisis_data.c  # Data processing
│   ├── include/          # Library headers
│   │   └── aisis.h       # Library API header
│   └── CMakeLists.txt    # Library build configuration
├── tests/                # Test files
├── docs/                 # Documentation
├── build/                # Build artifacts (generated)
├── bin/                  # Executables (generated)
├── lib/                  # Libraries (generated)
├── .vscode/              # VS Code configuration
├── CMakeLists.txt        # Main CMake configuration
├── Makefile              # Alternative build system
├── vcpkg-configuration.json # Package manager configuration
└── README.md             # This file
```

## Dependencies

### Required Tools
- **CMake** 3.20 or higher
- **GCC** or **Clang** compiler
- **Make** (for Makefile build)
- **ARM toolchain** (for embedded targets)

### vcpkg Packages (ARM Development)
- `arm:tools/open-cmsis-pack/cmsis-toolbox`
- `arm:compilers/arm/armclang`
- `arm:debuggers/arm/armdbg`
- `arm:tools/arm/mdk-toolbox`
- `arm:models/arm/avh-fvp`
- `arm:tools/kitware/cmake`
- `arm:tools/ninja-build/ninja`

## Building

### Using CMake (Recommended)

```bash
# Create build directory
mkdir build && cd build

# Configure the project
cmake ..

# Build the project
cmake --build .

# Run the executable
./bin/aisis_main
```

### Using Make

```bash
# Build everything
make

# Build with debug symbols
make debug

# Run the executable
make run

# Clean build artifacts
make clean
```

### Build Options

- **Debug Build**: `cmake -DCMAKE_BUILD_TYPE=Debug ..`
- **Release Build**: `cmake -DCMAKE_BUILD_TYPE=Release ..`
- **ARM Target**: Configure ARM toolchain in CMake

## Usage

### Basic Example

```c
#include "aisis.h"

int main() {
    // Initialize AISIS library
    if (aisis_init() != AISIS_SUCCESS) {
        return -1;
    }
    
    // Scan for devices
    aisis_device_info_t devices[AISIS_MAX_DEVICES];
    int device_count = aisis_scan_devices(devices, AISIS_MAX_DEVICES);
    
    if (device_count > 0) {
        // Connect to first device
        aisis_connect_device(devices[0].device_id);
        
        // Process some data
        uint8_t input[] = {0x01, 0x02, 0x03, 0x04};
        uint8_t output[16];
        int processed = aisis_process_data(input, sizeof(input), 
                                         output, sizeof(output));
        
        // Disconnect
        aisis_disconnect_device();
    }
    
    // Cleanup
    aisis_cleanup();
    return 0;
}
```

### Advanced Configuration

```c
// Custom configuration
aisis_config_t config = {
    .buffer_size = 2048,
    .timeout_ms = 10000,
    .debug_mode = true,
    .device_name = "MyDevice"
};

// Initialize with custom config
aisis_init_with_config(&config);
```

## API Reference

### Core Functions
- `aisis_init()` - Initialize library with default settings
- `aisis_init_with_config()` - Initialize with custom configuration
- `aisis_run()` - Run main processing loop
- `aisis_cleanup()` - Cleanup and shutdown
- `aisis_get_version()` - Get library version
- `aisis_get_status()` - Get current status

### Device Management
- `aisis_scan_devices()` - Scan for available devices
- `aisis_connect_device()` - Connect to specific device
- `aisis_disconnect_device()` - Disconnect from current device

### Data Processing
- `aisis_process_data()` - Process data buffer
- `aisis_send_command()` - Send command to device
- `aisis_read_response()` - Read response from device

## Error Codes

- `AISIS_SUCCESS` (0) - Operation successful
- `AISIS_ERROR_INIT` (-1) - Initialization error
- `AISIS_ERROR_INVALID` (-2) - Invalid parameter
- `AISIS_ERROR_MEMORY` (-3) - Memory allocation error
- `AISIS_ERROR_IO` (-4) - I/O operation error
- `AISIS_ERROR_TIMEOUT` (-5) - Operation timeout

## Development

### Adding New Features

1. Add function declarations to `include/aisis.h` or `aisis/include/aisis.h`
2. Implement functions in appropriate source files in `aisis/src/`
3. Update CMakeLists.txt if needed
4. Add tests in `tests/` directory
5. Update documentation

### Coding Standards

- Use C11 standard for C code
- Use C++17 standard for C++ code
- Follow consistent naming conventions
- Add comprehensive documentation
- Include error handling
- Write unit tests for new features

### VS Code Configuration

The project includes VS Code configuration files:
- `.vscode/c_cpp_properties.json` - IntelliSense configuration
- `.vscode/launch.json` - Debug configuration
- `.vscode/settings.json` - Editor settings

## Testing

```bash
# Build and run tests
cd build
cmake --build . --target test

# Or using Make
make test
```

## Installation

### System Installation

```bash
# Using Make
sudo make install

# Manual installation
sudo cp bin/aisis_main /usr/local/bin/
sudo cp lib/libaisis.a /usr/local/lib/
sudo cp aisis/include/aisis.h /usr/local/include/
```

### Uninstallation

```bash
sudo make uninstall
```

## Troubleshooting

### Common Issues

1. **Build Errors**: Ensure all dependencies are installed
2. **ARM Toolchain**: Verify vcpkg configuration and ARM tools
3. **Permission Errors**: Use sudo for system installation
4. **Missing Headers**: Check include paths in build configuration

### Debug Mode

Enable debug mode for verbose output:

```c
aisis_config_t config = {
    .debug_mode = true,
    // ... other settings
};
aisis_init_with_config(&config);
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Version History

- **v1.0.0** - Initial release with core functionality
  - Device management
  - Data processing
  - Command interface
  - ARM toolchain support

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the API reference above