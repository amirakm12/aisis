# CMakeLists.txt for Ultimate System
cmake_minimum_required(VERSION 3.16)

# Project configuration
project(UltimateSystem 
    VERSION 1.0.0
    DESCRIPTION "Ultimate AI-Powered System with Warlord Performance Mode"
    LANGUAGES CXX C
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform-specific configurations
if(WIN32)
    set(CMAKE_SYSTEM_VERSION 10.0)
    add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX)
    
    # Enable maximum performance optimizations for Windows
    if(MSVC)
        # MSVC compiler flags for maximum performance
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /Oi /Ot /Oy /GL /DNDEBUG /arch:AVX2")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
        
        # Enable all CPU optimizations
        add_compile_options(/favor:INTEL64)
        add_compile_options(/fp:fast)
        
        # Warlord mode specific flags
        add_compile_definitions(WARLORD_MODE=1)
        
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # GCC/Clang flags for maximum performance
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -mavx2 -mfma")
        add_compile_options(-ffast-math -funroll-loops -flto)
        
        # Enable all available optimizations
        add_compile_options(-fomit-frame-pointer -finline-functions)
        
        # Warlord mode specific flags
        add_compile_definitions(WARLORD_MODE=1)
    endif()
    
    # Windows-specific libraries
    set(PLATFORM_LIBS powrprof winmm)
    
elseif(UNIX AND NOT APPLE)
    # Linux optimizations (fallback for cross-platform compatibility)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
        add_compile_options(-ffast-math -funroll-loops)
    endif()
    set(PLATFORM_LIBS pthread)
endif()

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files for main application
set(MAIN_SOURCES
    src/main.cpp
    src/UltimateApplication.cpp
)

# Source files for Warlord Performance System
set(WARLORD_SOURCES
    src/warlord_performance.cpp
)

# AI-related source files
file(GLOB_RECURSE AI_SOURCES 
    "src/ai/*.cpp"
    "src/ai/*.c"
)

# Header files
file(GLOB_RECURSE HEADERS 
    "include/*.h"
    "include/*.hpp"
    "src/*.h"
    "src/*.hpp"
)

# Create main executable
add_executable(UltimateSystem ${MAIN_SOURCES} ${AI_SOURCES} ${HEADERS})

# Create Warlord Performance executable (Windows only)
if(WIN32)
    add_executable(WarlordPerformance ${WARLORD_SOURCES})
    
    # Warlord-specific compiler settings
    target_compile_options(WarlordPerformance PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2 /favor:INTEL64 /fp:fast>
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-mavx2 -mfma -ffast-math>
    )
    
    # Link Windows-specific libraries for Warlord
    target_link_libraries(WarlordPerformance PRIVATE 
        ${PLATFORM_LIBS}
        Threads::Threads
    )
    
    # Set Warlord executable properties
    set_target_properties(WarlordPerformance PROPERTIES
        OUTPUT_NAME "warlord_performance"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    
    # Enable Large Address Aware for 32-bit builds
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set_target_properties(WarlordPerformance PROPERTIES
            LINK_FLAGS "/LARGEADDRESSAWARE"
        )
    endif()
    
    # Add Windows manifest for high DPI awareness and admin privileges
    if(MSVC)
        target_sources(WarlordPerformance PRIVATE "${CMAKE_SOURCE_DIR}/resources/warlord.manifest")
    endif()
endif()

# Link libraries for main application
target_link_libraries(UltimateSystem PRIVATE 
    ${PLATFORM_LIBS}
    Threads::Threads
)

# Set executable properties
set_target_properties(UltimateSystem PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Compiler-specific optimizations
if(MSVC)
    # Enable all optimizations for release builds
    target_compile_options(UltimateSystem PRIVATE
        $<$<CONFIG:Release>:/O2 /Ob2 /Oi /Ot /Oy /GL>
        $<$<CONFIG:Release>:/arch:AVX2>
        /W4
    )
    
    # Enable Link Time Code Generation
    set_target_properties(UltimateSystem PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(UltimateSystem PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native>
        $<$<CONFIG:Release>:-ffast-math -funroll-loops>
        -Wall -Wextra -Wpedantic
    )
    
    # Enable Link Time Optimization
    set_target_properties(UltimateSystem PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()

# Build type specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "🔥 BUILDING IN WARLORD PERFORMANCE MODE 🔥")
    add_compile_definitions(NDEBUG WARLORD_PERFORMANCE_BUILD)
    
    # Strip symbols in release builds (non-MSVC)
    if(NOT MSVC)
        set_target_properties(UltimateSystem PROPERTIES
            LINK_FLAGS_RELEASE "-s"
        )
        if(WIN32)
            set_target_properties(WarlordPerformance PROPERTIES
                LINK_FLAGS_RELEASE "-s"
            )
        endif()
    endif()
    
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "🐛 Debug build - performance optimizations disabled")
    add_compile_definitions(_DEBUG DEBUG_BUILD)
    
    # Debug-specific flags
    if(MSVC)
        target_compile_options(UltimateSystem PRIVATE /Od /Zi)
    else()
        target_compile_options(UltimateSystem PRIVATE -O0 -g)
    endif()
endif()

# Custom build targets
if(WIN32)
    # Custom target to run system optimization scripts
    add_custom_target(optimize_system
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/windows_warlord_setup.bat
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "🔥 Running Warlord system optimizations..."
    )
    
    # Custom target for advanced tuning
    add_custom_target(advanced_tuning
        COMMAND powershell -ExecutionPolicy Bypass -File ${CMAKE_SOURCE_DIR}/scripts/advanced_warlord_tuning.ps1
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "⚡ Running advanced Warlord tuning..."
    )
    
    # Custom target to build and deploy everything
    add_custom_target(warlord_deploy
        DEPENDS WarlordPerformance optimize_system
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                $<TARGET_FILE:WarlordPerformance> 
                ${CMAKE_SOURCE_DIR}/bin/
        COMMENT "🚀 Deploying Warlord Performance System..."
    )
endif()

# Installation rules
install(TARGETS UltimateSystem
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(WIN32)
    install(TARGETS WarlordPerformance
        RUNTIME DESTINATION bin
    )
    
    # Install scripts
    install(FILES 
        ${CMAKE_SOURCE_DIR}/scripts/windows_warlord_setup.bat
        ${CMAKE_SOURCE_DIR}/scripts/advanced_warlord_tuning.ps1
        DESTINATION scripts
    )
endif()

# Install documentation
install(FILES 
    ${CMAKE_SOURCE_DIR}/docs/WARLORD_PERFORMANCE_GUIDE.md
    ${CMAKE_SOURCE_DIR}/README.md
    DESTINATION docs
)

# CPack configuration for creating installers
set(CPACK_PACKAGE_NAME "UltimateSystem")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Ultimate AI System with Warlord Performance Mode")
set(CPACK_PACKAGE_VENDOR "Warlord Performance Labs")

if(WIN32)
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_NSIS_DISPLAY_NAME "Ultimate System - Warlord Edition")
    set(CPACK_NSIS_HELP_LINK "https://github.com/your-repo/ultimate-system")
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
endif()

include(CPack)

# Print build information
message(STATUS "")
message(STATUS "🔥🔥🔥 ULTIMATE SYSTEM BUILD CONFIGURATION 🔥🔥🔥")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

if(WIN32)
    message(STATUS "🎯 Warlord Performance Mode: ENABLED")
    message(STATUS "💀 AVX2 Optimizations: ENABLED")
    message(STATUS "⚡ Maximum Performance Flags: ACTIVE")
else()
    message(STATUS "⚠️ Warlord Performance Mode: LIMITED (Windows only)")
endif()

message(STATUS "🔥🔥🔥 BUILD READY FOR SYSTEM DOMINATION 🔥🔥🔥")
message(STATUS "") 