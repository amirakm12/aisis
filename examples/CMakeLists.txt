# Examples CMakeLists.txt

# AI Acceleration Demo
add_executable(ai_acceleration_demo ai_acceleration_demo.cpp)
target_link_libraries(ai_acceleration_demo ultimate_core)

# Set output directory
set_target_properties(ai_acceleration_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Basic usage example
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/basic_usage.cpp")
    add_executable(basic_usage basic_usage.cpp)
    target_link_libraries(basic_usage ultimate_core)
    
    set_target_properties(basic_usage PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Print examples information
message(STATUS "Building examples:")
message(STATUS "  - ai_acceleration_demo")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/basic_usage.cpp")
    message(STATUS "  - basic_usage")
endif() 