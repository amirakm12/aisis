cmake_minimum_required(VERSION 3.20)

# Project name and version
project(aisis VERSION 1.0.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Export compile commands for better IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Option to build for ARM
option(BUILD_FOR_ARM "Build for ARM Cortex-M target" OFF)

if(BUILD_FOR_ARM)
    # ARM specific settings
    set(CMAKE_SYSTEM_NAME Generic)
    set(CMAKE_SYSTEM_PROCESSOR arm)
    
    # Specify the cross compiler
    set(CMAKE_C_COMPILER arm-none-eabi-gcc)
    set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
    set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
    set(CMAKE_AR arm-none-eabi-ar)
    set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
    set(CMAKE_OBJDUMP arm-none-eabi-objdump)
    set(SIZE arm-none-eabi-size)
    
    # ARM compiler flags
    set(MCU_FLAGS "-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16")
    set(CMAKE_C_FLAGS "${MCU_FLAGS} -fdata-sections -ffunction-sections -Wall -fstack-usage")
    set(CMAKE_CXX_FLAGS "${MCU_FLAGS} -fdata-sections -ffunction-sections -Wall -fstack-usage")
    
    # Linker flags
    set(CMAKE_EXE_LINKER_FLAGS "${MCU_FLAGS} -specs=nano.specs -Wl,--gc-sections -Wl,--print-memory-usage")
else()
    # Native build settings
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic -g -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -g -O0")
endif()

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/aisis
)

# Find packages (if using vcpkg)
if(DEFINED ENV{VCPKG_ROOT})
    include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
endif()

# Collect source files
file(GLOB_RECURSE SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/aisis/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aisis/*.c
)

# Create executable
if(BUILD_FOR_ARM)
    add_executable(${PROJECT_NAME}.elf ${SOURCES})
    
    # Generate hex and bin files
    add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -Oihex $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.hex
        COMMAND ${CMAKE_OBJCOPY} -Obinary $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.bin
        COMMENT "Building ${PROJECT_NAME}.hex and ${PROJECT_NAME}.bin"
    )
else()
    add_executable(${PROJECT_NAME} ${SOURCES})
endif()

# Enable testing
enable_testing()

# Add test subdirectory if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_subdirectory(tests)
endif()

# Installation rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
if(BUILD_FOR_ARM)
    message(STATUS "Building for ARM Cortex-M4")
endif()