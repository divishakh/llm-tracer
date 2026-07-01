#include "ggml_hooks.hpp"
#include "llama.h"
#include "ggml.h"
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct HookContext {
    RingBuffer<LayerEvent, 512>* buffer;
    int num_tokens;
    std::string token_strs[ATTN_SIZE];
    std::chrono::high_resolution_clock::time_point layer_start;
    LayerEvent pending;
    bool layer_open;
};

HookContext* g_ctx = nullptr;

bool starts_with(const char* str, const char* prefix) {
    return strncmp(str, prefix, strlen(prefix)) == 0;
}

int extract_layer_index(const char* name) {
    const char* dash = strrchr(name, '-');
    if (!dash) return -1;
    return atoi(dash + 1);
}

float tensor_mean(const ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32) return 0.0f;
    int64_t n = ggml_nelements(t);
    if (n == 0) return 0.0f;
    const float* data = (const float*)t->data;
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) sum += data[i];
    return (float)(sum / n);
}

float tensor_max_abs(const ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32) return 0.0f;
    int64_t n = ggml_nelements(t);
    const float* data = (const float*)t->data;
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float v = fabsf(data[i]);
        if (v > m) m = v;
    }
    return m;
}

float tensor_sparsity_pct(const ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32) return 0.0f;
    int64_t n = ggml_nelements(t);
    if (n == 0) return 0.0f;
    const float* data = (const float*)t->data;
    int64_t zero_count = 0;
    for (int64_t i = 0; i < n; i++) {
        if (fabsf(data[i]) < 1e-6f) zero_count++;
    }
    return 100.0f * (float)zero_count / (float)n;
}

bool real_model_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    if (ask) return true;
    if (g_ctx == nullptr) return true;

    const char* name = t->name;

    // Start of a new layer
    if (starts_with(name, "attn_norm-")) {
        int idx = extract_layer_index(name);
        if (idx >= 0) {
            g_ctx->layer_open  = true;
            g_ctx->layer_start = std::chrono::high_resolution_clock::now();
            g_ctx->pending = LayerEvent{};
            g_ctx->pending.layer_index = idx;
            g_ctx->pending.layer_type  = "Attn (Self)";
            g_ctx->pending.device      = "CPU (Real Model)";
            g_ctx->pending.num_tokens  = g_ctx->num_tokens;
            for (int i = 0; i < ATTN_SIZE; i++)
                g_ctx->pending.tokens[i] = g_ctx->token_strs[i];
        }
    }

    // Real attention weights from softmax(QK^T), head 0
    if (g_ctx->layer_open && starts_with(name, "kq_soft_max-")) {
        if (t->type == GGML_TYPE_F32) {
            int64_t n_keys  = t->ne[0];
            int64_t n_query = t->ne[1];
            const float* data = (const float*)t->data;
            int limit = ATTN_SIZE;
            if (limit > n_query) limit = (int)n_query;

            for (int q = 0; q < ATTN_SIZE; q++) {
                for (int k = 0; k < ATTN_SIZE; k++) {
                    float v = 0.0f;
                    if (q < limit && k < limit) {
                        int64_t offset = k + q * n_keys; // head 0
                        v = data[offset];
                    }
                    g_ctx->pending.attn_matrix[q][k] = v;
                }
            }
        }
    }

    // End of layer - capture stats from this layer's final output
    if (g_ctx->layer_open && starts_with(name, "l_out-")) {
        int idx = extract_layer_index(name);
        if (idx == g_ctx->pending.layer_index) {
            g_ctx->pending.mean     = tensor_mean(t);
            g_ctx->pending.max_val  = tensor_max_abs(t);
            g_ctx->pending.sparsity = tensor_sparsity_pct(t);
            g_ctx->pending.shape[0] = (int)t->ne[0];
            g_ctx->pending.shape[1] = (int)t->ne[1];
            g_ctx->pending.shape[2] = (int)t->ne[2];

            auto now = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(
                           now - g_ctx->layer_start);
            g_ctx->pending.latency_ms = dur.count() / 1000.0f;

            auto sys_now = std::chrono::system_clock::now();
            g_ctx->pending.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            sys_now.time_since_epoch()).count();

            g_ctx->buffer->push(g_ctx->pending);
            g_ctx->layer_open = false;
        }
    }

    return true;
}

} // namespace

bool GGMLHooks::run_real_model(RingBuffer<LayerEvent, 512>& buffer,
                                const std::string& model_path,
                                const std::string& prompt) {
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        printf("Failed to load model: %s\n", model_path.c_str());
        return false;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(256);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                   tokens.data(), (int)tokens.size(),
                                   true, false);
    if (n_tokens <= 0) {
        printf("Tokenization failed.\n");
        llama_model_free(model);
        return false;
    }
    tokens.resize(n_tokens);

    HookContext ctx;
    ctx.buffer     = &buffer;
    ctx.num_tokens = n_tokens;
    ctx.layer_open = false;

    char piece_buf[64];
    for (int i = 0; i < ATTN_SIZE; i++) {
        if (i < n_tokens) {
            int len = llama_token_to_piece(vocab, tokens[i], piece_buf,
                                            sizeof(piece_buf), 0, false);
            ctx.token_strs[i] = (len > 0) ? std::string(piece_buf, len) : "?";
        } else {
            ctx.token_strs[i] = "";
        }
    }

    g_ctx = &ctx;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.cb_eval = real_model_callback;
    ctx_params.cb_eval_user_data = nullptr;

    llama_context* lctx = llama_init_from_model(model, ctx_params);
    if (lctx == nullptr) {
        printf("Failed to create context.\n");
        llama_model_free(model);
        g_ctx = nullptr;
        return false;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    int rc = llama_decode(lctx, batch);

    g_ctx = nullptr;

    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();

    return rc == 0;
}

void GGMLHooks::simulate_forward_pass(RingBuffer<LayerEvent, 512>& buffer) {
    const char* layer_types[] = {"Attn (Self)", "MLP (SwiGLU)", "LayerNorm"};
    const char* devices[]     = {"CUDA [GPU 0]", "CPU (Fallback)"};
    const char* sample_tokens[] = {"I","want","it","to","be","fast","and","good"};

    for (int i = 0; i < 12; i++) {
        LayerEvent ev{};
        ev.layer_index = i;
        ev.layer_type  = layer_types[i % 3];
        ev.device      = (i == 5) ? devices[1] : devices[0];
        ev.latency_ms  = 0.5f + (float)(rand() % 100) / 100.0f;
        ev.mean        = (float)(rand() % 100) / 100.0f;
        ev.max_val     = 3.0f + (float)(rand() % 400) / 100.0f;
        ev.sparsity    = (float)(rand() % 100);
        ev.shape[0] = 1; ev.shape[1] = 32; ev.shape[2] = 4096;
        ev.num_tokens = ATTN_SIZE;
        for (int t = 0; t < ATTN_SIZE; t++) ev.tokens[t] = sample_tokens[t];
        for (int row = 0; row < ATTN_SIZE; row++) {
            float row_sum = 0.0f;
            for (int col = 0; col < ATTN_SIZE; col++) {
                float dist = (float)abs(row - col);
                float val = expf(-dist * 0.8f) + (float)(rand()%20)/100.0f;
                if (col > row) val = 0.0f;
                ev.attn_matrix[row][col] = val;
                row_sum += val;
            }
            if (row_sum > 0)
                for (int col = 0; col < ATTN_SIZE; col++)
                    ev.attn_matrix[row][col] /= row_sum;
        }
        auto now = std::chrono::system_clock::now();
        ev.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now.time_since_epoch()).count();
        buffer.push(ev);
    }
}