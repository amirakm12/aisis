cmake_minimum_required(VERSION 3.20)

# Set the project name and version
project(aisis VERSION 1.0.0 LANGUAGES C CXX)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options for ARM development
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    add_compile_options(-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16)
    add_link_options(-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16)
endif()

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Add compiler flags
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/aisis/include
    ${CMAKE_CURRENT_SOURCE_DIR}/aisis/src
)

# Add subdirectories
add_subdirectory(aisis)

# Create the main executable
add_executable(${PROJECT_NAME}_main
    src/main.c
)

# Link libraries
target_link_libraries(${PROJECT_NAME}_main
    aisis_lib
)

# Set output directory
set_target_properties(${PROJECT_NAME}_main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable testing
enable_testing()

# Add test executable
add_executable(test_aisis
    tests/test_aisis.c
)

# Link test executable with the library
target_link_libraries(test_aisis
    aisis_lib
)

# Set test output directory
set_target_properties(test_aisis PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the test
add_test(NAME aisis_tests COMMAND test_aisis)

# Add custom target for cleaning
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
)