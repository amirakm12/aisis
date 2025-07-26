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

# Cross-compilation support
if(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling for ${CMAKE_SYSTEM_NAME}")
endif()

# Windows-specific settings
if(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0601)  # Windows 7 or later
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DUNICODE -D_UNICODE)
    
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
        psapi
        d3d11
        dxgi
        opengl32
        glu32
    )
    
    # MinGW-specific settings
    if(MINGW)
        message(STATUS "Building with MinGW")
        add_compile_options(-municode)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -municode")
    endif()
endif()

# Linux-specific settings
if(UNIX AND NOT APPLE)
    set(LINUX_LIBRARIES
        pthread
        m
        dl
        rt
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
        /wd4244  # Conversion warnings
        /wd4267  # Size_t conversion warnings
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi /RTC1 /MDd)
        add_definitions(-DDEBUG=1 -DULTIMATE_DEBUG_ENABLED=1)
    else()
        add_compile_options(/O2 /Ob2 /Oi /Ot /MD)
        add_definitions(-DNDEBUG=1 -DULTIMATE_DEBUG_ENABLED=0)
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g3)
        add_definitions(-DDEBUG=1 -DULTIMATE_DEBUG_ENABLED=1)
    else()
        add_compile_options(-O2 -g1)
        add_definitions(-DNDEBUG=1 -DULTIMATE_DEBUG_ENABLED=0)
    endif()
    
    # Enable position independent code
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
    ${CMAKE_CURRENT_SOURCE_DIR}/include/graphics
    ${CMAKE_CURRENT_SOURCE_DIR}/include/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/include/neural
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ai
    ${CMAKE_CURRENT_SOURCE_DIR}/include/reality
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

file(GLOB_RECURSE GRAPHICS_SOURCES
    "src/graphics/*.c"
    "src/graphics/*.cpp"
)

file(GLOB_RECURSE AUDIO_SOURCES
    "src/audio/*.c"
    "src/audio/*.cpp"
)

file(GLOB_RECURSE REALITY_SOURCES
    "src/reality/*.c"
    "src/reality/*.cpp"
)

# Combine all sources
set(ALL_SOURCES
    ${CORE_SOURCES}
    ${HARDWARE_SOURCES}
    ${NEURAL_SOURCES}
    ${AI_SOURCES}
    ${GRAPHICS_SOURCES}
    ${AUDIO_SOURCES}
    ${REALITY_SOURCES}
)

# Create the main library
add_library(ultimate_core STATIC ${ALL_SOURCES})

# Target properties
set_target_properties(ultimate_core PROPERTIES
    OUTPUT_NAME "ultimate"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Target compile definitions
target_compile_definitions(ultimate_core PRIVATE
    ULTIMATE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    ULTIMATE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    ULTIMATE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Link libraries based on platform
if(WIN32)
    target_link_libraries(ultimate_core PRIVATE ${WINDOWS_LIBRARIES})
elseif(UNIX AND NOT APPLE)
    target_link_libraries(ultimate_core PRIVATE ${LINUX_LIBRARIES})
endif()

# Create shared library option
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
if(BUILD_SHARED_LIBS)
    add_library(ultimate_core_shared SHARED ${ALL_SOURCES})
    set_target_properties(ultimate_core_shared PROPERTIES
        OUTPUT_NAME "ultimate"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    
    target_compile_definitions(ultimate_core_shared PRIVATE
        ULTIMATE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        ULTIMATE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        ULTIMATE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    )
    
    if(WIN32)
        target_link_libraries(ultimate_core_shared PRIVATE ${WINDOWS_LIBRARIES})
    elseif(UNIX AND NOT APPLE)
        target_link_libraries(ultimate_core_shared PRIVATE ${LINUX_LIBRARIES})
    endif()
endif()

# Create main executable
add_executable(ultimate_app src/main.cpp src/UltimateApplication.cpp)
target_link_libraries(ultimate_app PRIVATE ultimate_core)

# Set executable properties
set_target_properties(ultimate_app PROPERTIES
    OUTPUT_NAME "ultimate"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS ultimate_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

if(BUILD_SHARED_LIBS)
    install(TARGETS ultimate_core_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(TARGETS ultimate_app
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Optional: Create example executables
option(BUILD_EXAMPLES "Build example applications" ON)
if(BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    add_subdirectory(examples)
endif()

# Optional: Create tests
option(BUILD_TESTS "Build test suite" OFF)
if(BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        add_subdirectory(tests)
    else()
        message(WARNING "BUILD_TESTS enabled but tests directory not found")
    endif()
endif()

# Optional: Create benchmarks
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
if(BUILD_BENCHMARKS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
        add_subdirectory(benchmarks)
    else()
        message(WARNING "BUILD_BENCHMARKS enabled but benchmarks directory not found")
    endif()
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "ULTIMATE_System")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Universal Low-latency Technology for Intelligent Memory and Task Execution")
set(CPACK_PACKAGE_VENDOR "ULTIMATE Systems")
set(CPACK_PACKAGE_CONTACT "support@ultimate-systems.com")

if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "ULTIMATE System")
    set(CPACK_NSIS_PACKAGE_NAME "ULTIMATE System")
    set(CPACK_NSIS_HELP_LINK "https://ultimate-systems.com")
    set(CPACK_NSIS_URL_INFO_ABOUT "https://ultimate-systems.com")
    set(CPACK_NSIS_CONTACT "support@ultimate-systems.com")
else()
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "ULTIMATE Systems <support@ultimate-systems.com>")
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
endif()

include(CPack)

# Print build information
message(STATUS "ULTIMATE System Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  CXX Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Build Shared Libraries: ${BUILD_SHARED_LIBS}")
if(CMAKE_CROSSCOMPILING)
    message(STATUS "  Cross-compiling: YES")
    message(STATUS "  Target system: ${CMAKE_SYSTEM_NAME}")
endif()

# Add custom targets for development
add_custom_target(count_functions
    COMMAND find include -name "*.h" -exec grep -H "(" {} "\;" | grep -E "(ultimate_|ULTIMATE_)" | wc -l
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Counting ULTIMATE API functions"
)

add_custom_target(clean_build
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMENT "Cleaning build directory"
)

# Windows-specific build tools setup
if(WIN32 AND NOT CMAKE_CROSSCOMPILING)
    # Check for Visual Studio or MinGW
    if(MSVC)
        message(STATUS "Using Visual Studio compiler")
    elseif(MINGW)
        message(STATUS "Using MinGW compiler")
        # Add MinGW-specific optimizations
        add_compile_options(-march=native -mtune=native)
    else()
        message(WARNING "Unknown Windows compiler - build may fail")
    endif()
endif() 