#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <string>

#define ATTN_SIZE 8

struct LayerEvent {
    int         layer_index;
    std::string layer_type;
    std::string device;
    float       latency_ms;
    float       mean;
    float       max_val;
    float       sparsity;
    int         shape[3];
    long long   timestamp;

    float       attn_matrix[ATTN_SIZE][ATTN_SIZE];
    std::string tokens[ATTN_SIZE];
    int         num_tokens;
};

template<typename T, size_t N>
class RingBuffer {
    std::array<T, N>    buf;
    std::atomic<size_t> head{0};
    std::atomic<size_t> tail{0};

public:
    bool push(T item) {
        size_t h    = head.load();
        size_t next = (h + 1) % N;
        if (next == tail.load()) return false;
        buf[h] = item;
        head.store(next);
        return true;
    }

    bool pop(T& out) {
        size_t t = tail.load();
        if (t == head.load()) return false;
        out = buf[t];
        tail.store((t + 1) % N);
        return true;
    }

    bool empty() const {
        return head.load() == tail.load();
    }
};