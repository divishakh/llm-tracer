#pragma once
#include "ring_buffer.hpp"
#include <string>

class GGMLHooks {
public:
    // Loads a real GGUF model, runs the given prompt through it,
    // and pushes real LayerEvent data into the ring buffer.
    static bool run_real_model(RingBuffer<LayerEvent, 512>& buffer,
                                const std::string& model_path,
                                const std::string& prompt);

    // Fallback fake data generator (kept for safety)
    static void simulate_forward_pass(RingBuffer<LayerEvent, 512>& buffer);
};