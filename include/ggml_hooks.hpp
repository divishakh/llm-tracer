#pragma once
#include "ring_buffer.hpp"
#include <functional>

// This class hooks into the model's execution
// without modifying the model's code
class GGMLHooks {
public:
    // Start hooking into the model
    static void attach(RingBuffer<LayerEvent, 512>& buffer);
    
    // Stop hooking
    static void detach();

    // Simulate a fake forward pass for testing
    static void simulate_forward_pass(RingBuffer<LayerEvent, 512>& buffer);

private:
    static bool is_attached;
};