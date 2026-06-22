#include "ggml_hooks.hpp"
#include <chrono>
#include <cstdlib>

bool GGMLHooks::is_attached = false;

void GGMLHooks::attach(RingBuffer<LayerEvent, 512>& buffer) {
    is_attached = true;
}

void GGMLHooks::detach() {
    is_attached = false;
}

// This simulates fake layer events so we can test the TUI
// Later this will be replaced with real model hooks
void GGMLHooks::simulate_forward_pass(RingBuffer<LayerEvent, 512>& buffer) {
    const char* layer_types[] = {"Attn (Self)", "MLP (SwiGLU)", "LayerNorm"};
    const char* devices[]     = {"CUDA [GPU 0]", "CPU (Fallback)"};

    for (int i = 0; i < 12; i++) {
        LayerEvent ev;
        ev.layer_index = i;
        ev.layer_type  = layer_types[i % 3];
        ev.device      = (i == 5) ? devices[1] : devices[0];
        ev.latency_ms  = 0.5f + (float)(rand() % 100) / 100.0f;
        ev.mean        = (float)(rand() % 100) / 100.0f;
        ev.max_val     = 3.0f + (float)(rand() % 400) / 100.0f;
        ev.sparsity    = (float)(rand() % 100);
        ev.shape[0]    = 1;
        ev.shape[1]    = 32;
        ev.shape[2]    = 4096;

        auto now       = std::chrono::system_clock::now();
        ev.timestamp   = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()).count();

        buffer.push(ev);
    }
}