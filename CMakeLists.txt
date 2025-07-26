cmake_minimum_required(VERSION 3.20)
project(VectorDB VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Check for OpenMP support
find_package(OpenMP)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/vector_db.cpp
    src/embedding.cpp
    src/index.cpp
    src/similarity.cpp
    src/storage.cpp
)

# Header files
set(HEADERS
    include/vector_db.h
    include/embedding.h
    include/index.h
    include/similarity.h
    include/storage.h
    include/types.h
)

# Create the library
add_library(vectordb STATIC ${SOURCES} ${HEADERS})

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(vectordb PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(vectordb PUBLIC USE_OPENMP)
endif()

# Create example executable
add_executable(vectordb_example examples/main.cpp)
target_link_libraries(vectordb_example vectordb)

# Create test executable
add_executable(vectordb_test tests/test_main.cpp)
target_link_libraries(vectordb_test vectordb)

# Installation
install(TARGETS vectordb vectordb_example
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(FILES ${HEADERS} DESTINATION include/vectordb)