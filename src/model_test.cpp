#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

// This callback fires for EVERY tensor operation during inference.
// ask=true  -> llama.cpp is asking "do you want data for this tensor?"
// ask=false -> the tensor's data is now ready, this is where we read it
static bool my_tensor_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    if (ask) {
        // Return true to say "yes, give me this tensor's data"
        return true;
    }

    // At this point t->data contains real numbers from the model
    const char* name = t->name;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];

    printf("[TENSOR] name=%-30s shape=[%lld,%lld,%lld]\n",
           name, (long long)ne0, (long long)ne1, (long long)ne2);

    return true;
}

int main() {
    const char* model_path = "../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    llama_backend_init();

    printf("Loading model...\n");
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path, model_params);

    if (model == nullptr) {
        printf("FAILED to load model!\n");
        return 1;
    }
    printf("Model loaded.\n");

    // Set up context with our callback attached
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;          // max context window for this test
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.cb_eval = my_tensor_callback;
    ctx_params.cb_eval_user_data = nullptr;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        printf("FAILED to create context!\n");
        return 1;
    }
    printf("Context created with callback attached.\n");

    // Tokenize a simple prompt
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const char* prompt = "The capital of France is";

    std::vector<llama_token> tokens(64);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                   tokens.data(), tokens.size(),
                                   true, false);
    tokens.resize(n_tokens);

    printf("Prompt tokenized into %d tokens.\n", n_tokens);
    printf("Running inference (callback will fire for each tensor)...\n\n");

    // Run one forward pass - this triggers our callback repeatedly
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        printf("FAILED to decode!\n");
        return 1;
    }

    printf("\nInference complete!\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}