cmake_minimum_required(VERSION 3.20)

# Set the project name and version
project(ULTIMATE_System VERSION 1.0.0 LANGUAGES C CXX)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Windows-specific settings
if(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0601)  # Windows 7 or later
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    
    # Set Windows subsystem
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
    
    # Windows libraries
    set(WINDOWS_LIBRARIES
        kernel32
        user32
        gdi32
        winspool
        shell32
        ole32
        oleaut32
        uuid
        comdlg32
        advapi32
        ws2_32
        winmm
        imm32
        oleacc
        version
    )
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(
        /W4
        /WX
        /wd4100  # Unreferenced parameter
        /wd4201  # Nameless struct/union
        /wd4996  # Deprecated functions
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi /RTC1)
        add_definitions(-DDEBUG=1)
    else()
        add_compile_options(/O2 /Ob2 /Oi /Ot)
        add_definitions(-DNDEBUG=1)
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g3)
        add_definitions(-DDEBUG=1)
    else()
        add_compile_options(-O2 -g1)
        add_definitions(-DNDEBUG=1)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
)

# Source files
file(GLOB_RECURSE CORE_SOURCES
    "src/core/*.c"
    "src/core/*.cpp"
)

file(GLOB_RECURSE HARDWARE_SOURCES
    "src/hardware/*.c"
    "src/hardware/*.cpp"
)

file(GLOB_RECURSE NEURAL_SOURCES
    "src/neural/*.c"
    "src/neural/*.cpp"
)

file(GLOB_RECURSE AI_SOURCES
    "src/ai/*.c"
    "src/ai/*.cpp"
)

# Create the main library
add_library(ultimate_core STATIC
    ${CORE_SOURCES}
    ${HARDWARE_SOURCES}
    ${NEURAL_SOURCES}
    ${AI_SOURCES}
)

# Target properties
set_target_properties(ultimate_core PROPERTIES
    OUTPUT_NAME "ultimate"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Target compile definitions
target_compile_definitions(ultimate_core PRIVATE
    ULTIMATE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    ULTIMATE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    ULTIMATE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Link Windows libraries
if(WIN32)
    target_link_libraries(ultimate_core PRIVATE ${WINDOWS_LIBRARIES})
endif()

# Installation
install(TARGETS ultimate_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Optional: Create example executables
option(BUILD_EXAMPLES "Build example applications" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Optional: Create tests
option(BUILD_TESTS "Build test suite" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print build information
message(STATUS "ULTIMATE System Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "  CXX Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Platform: Windows")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_TESTS}") 