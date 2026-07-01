cmake_minimum_required(VERSION 3.20)
project(llm_tracer CXX)
set(CMAKE_CXX_STANDARD 17)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/common
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)

# Force llama.cpp to build its common utils library
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

# Add FTXUI
add_subdirectory(third_party/ftxui)

# Add llama.cpp
add_subdirectory(third_party/llama.cpp)

# Our main executable
add_executable(llm_tracer
    src/main.cpp
    src/hooks/ggml_hooks.cpp
    src/capture/ring_buffer.cpp
    src/tui/app.cpp
    src/anomaly/detector.cpp
)

target_link_libraries(llm_tracer PRIVATE
    llama
    llama-common
    ftxui::screen
    ftxui::dom
    ftxui::component
)
# Standalone model loading test
add_executable(model_test
    src/model_test.cpp
)

target_link_libraries(model_test PRIVATE
    llama
)