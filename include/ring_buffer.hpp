#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <string>

// This struct holds information about each layer event
struct LayerEvent {
    int         layer_index;
    std::string layer_type;    // "Attention", "MLP", "LayerNorm"
    std::string device;        // "CUDA" or "CPU"
    float       latency_ms;    // how long this layer took
    float       mean;          // average activation value
    float       max_val;       // maximum activation value
    float       sparsity;      // percentage of zero values
    int         shape[3];      // tensor shape e.g. [1, 32, 4096]
    long long   timestamp;     // when this event happened
};

// Fixed size ring buffer - stores last 512 layer events
// Safe to use from two threads at the same time
template<typename T, size_t N>
class RingBuffer {
    std::array<T, N>    buf;
    std::atomic<size_t> head{0};
    std::atomic<size_t> tail{0};

public:
    // Add a new event (called from inference thread)
    bool push(T item) {
        size_t h = head.load();
        size_t next = (h + 1) % N;
        if (next == tail.load()) return false; // buffer full
        buf[h] = item;
        head.store(next);
        return true;
    }

    // Get next event (called from TUI thread)
    bool pop(T& out) {
        size_t t = tail.load();
        if (t == head.load()) return false; // buffer empty
        out = buf[t];
        tail.store((t + 1) % N);
        return true;
    }

    bool empty() const {
        return head.load() == tail.load();
    }
};