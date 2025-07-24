cmake_minimum_required(VERSION 3.31.5)
project(AISIS_Creative_Studio VERSION 2.0.0 LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Performance optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -flto -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

# Enable parallel compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# OpenCV for computer vision and image processing
find_package(OpenCV REQUIRED)

# OpenGL for graphics rendering
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

# Audio processing libraries
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
pkg_check_modules(FFTW REQUIRED fftw3)

# Networking
find_package(Boost REQUIRED COMPONENTS system filesystem thread asio)

# JSON processing
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${PORTAUDIO_INCLUDE_DIRS})
include_directories(${FFTW_INCLUDE_DIRS})

# Source files
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.c"
)

# Create executable
add_executable(aisis_studio ${SOURCES})

# Link libraries
target_link_libraries(aisis_studio
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    glfw
    GLEW::GLEW
    ${PORTAUDIO_LIBRARIES}
    ${FFTW_LIBRARIES}
    ${Boost_LIBRARIES}
    nlohmann_json::nlohmann_json
    Threads::Threads
    -fopenmp
)

# Compiler definitions for performance
target_compile_definitions(aisis_studio PRIVATE
    AISIS_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    AISIS_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    AISIS_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    AISIS_PERFORMANCE_MODE=1
    AISIS_MULTI_THREADING=1
    AISIS_GPU_ACCELERATION=1
)

# Create subdirectories
add_subdirectory(modules/graphics)
add_subdirectory(modules/audio)
add_subdirectory(modules/ai)
add_subdirectory(modules/networking)
add_subdirectory(modules/ui)

# Installation
install(TARGETS aisis_studio DESTINATION bin)
install(DIRECTORY assets/ DESTINATION share/aisis_studio/assets)
install(DIRECTORY config/ DESTINATION etc/aisis_studio)