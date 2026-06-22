cmake_minimum_required(VERSION 3.20)
project(llm_tracer CXX)
set(CMAKE_CXX_STANDARD 17)

# Tell compiler where to find our header files
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

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
    ftxui::screen
    ftxui::dom
    ftxui::component
)