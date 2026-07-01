# LLM Tracer

A lightweight, non-invasive instrumentation and tracing platform for local LLM inference, built in C++. It hooks into `llama.cpp`'s compute graph at runtime — without modifying any model code — to capture real layer-by-layer telemetry (latency, activation statistics, attention weights) and renders it live in a terminal UI (TUI).

Think of it as a heart-rate monitor for a language model: while a prompt runs through the model, you can watch each transformer layer's timing, output statistics, and self-attention pattern update in real time, with automatic flagging of numerical anomalies.

---

## Why this exists

Most LLM debugging tools are either:
- Python-based and slow to attach to a running C/C++ inference process, or
- Limited to static log dumps rather than live, navigable telemetry.

LLM Tracer instead hooks directly into `llama.cpp`'s `ggml` compute graph callback (`cb_eval`), giving microsecond-accurate per-tensor timing and direct memory access to real activation data — with zero changes to the model or `llama.cpp` source itself.

---

## Features

- **Non-invasive hooking** — Uses `llama.cpp`'s built-in `cb_eval` callback on the GGML compute graph. No fork, no patch, no recompiled model code.
- **Real per-layer telemetry** — For every transformer layer in the loaded model:
  - Latency (wall-clock time per layer, measured around the callback)
  - Mean activation value
  - Max absolute activation value
  - Activation sparsity (% of near-zero values)
  - Tensor shape
- **Real attention visualization** — Captures the actual `softmax(QK^T)` tensor (flash attention is disabled specifically so this tensor exists) and renders it as a colored block-character heatmap, token-by-token.
- **Adaptive anomaly detection** — Flags:
  - Outlier activations (configurable max-activation threshold)
  - Statistically anomalous layer latency (calibrated per-run via mean + 2σ, not a hardcoded number — adapts automatically to CPU vs GPU speed)
  - Device fallback events (e.g., a layer unexpectedly dropping from GPU to CPU)
- **Interactive 5-panel TUI** (built with [FTXUI](https://github.com/ArthurSonzogni/FTXUI)):
  1. Model topology / layer list (navigable)
  2. Live packet stream of captured layer events
  3. Attention matrix heatmap (toggle fullscreen)
  4. Runtime metrics inspector for the selected layer
  5. Anomaly ledger
- **Custom prompts** — Type any prompt at startup; the tracer tokenizes it and runs a real forward pass through the loaded GGUF model.
- **Fixed-memory ring buffer** — Lock-free SPSC ring buffer caps memory usage regardless of model size or run length.

### What makes this different from existing tools
Most "transformer visualizers" available online (BertViz, exBERT, attention-viz notebooks, etc.) are Python/Jupyter-based, operate on cached/offline activations, and are tied to HuggingFace `transformers`. LLM Tracer instead:
- Operates on **any GGUF model** runnable by `llama.cpp` (Llama, Mistral, Qwen, Phi, etc. — not limited to one architecture)
- Captures telemetry **during actual C++ inference**, not from a separate logging pass
- Runs as a **standalone terminal binary** with no Python/Jupyter/browser dependency
- Is built for **systems-level debugging** (latency, device fallback, memory) in addition to attention visualization

---

## Project Structure

```
llm-tracer/
├── CMakeLists.txt
├── include/
│   ├── ring_buffer.hpp       # LayerEvent struct + lock-free ring buffer
│   ├── ggml_hooks.hpp        # Real-model + simulated hook interfaces
│   ├── detector.hpp          # Anomaly struct + adaptive detector
│   └── tui/app.hpp           # TracerApp interface
├── src/
│   ├── main.cpp              # Entry point — prompts user, launches TracerApp
│   ├── hooks/ggml_hooks.cpp  # Real cb_eval callback + model loading/inference
│   ├── capture/ring_buffer.cpp
│   ├── anomaly/detector.cpp  # Anomaly checks (activation, latency, device)
│   └── tui/app.cpp           # FTXUI rendering, panels, keyboard handling
├── models/                   # GGUF model files (gitignored — see below)
└── third_party/
    ├── llama.cpp             # Git submodule — inference backend
    └── ftxui                 # Git submodule — terminal UI library
```

---

## Requirements

- Windows 10/11 (instructions below use MSYS2/MinGW; Linux/macOS work with system GCC/Clang + CMake equivalently)
- [MSYS2](https://www.msys2.org) with the UCRT64 toolchain (`gcc`, `cmake`, `ninja`)
- Git
- ~700 MB free disk space for the test GGUF model
- A GGUF-format language model (this project was tested with [TinyLlama-1.1B-Chat-v1.0 Q4_K_M](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF))

---

## Setup & Build Instructions

### 1. Clone the repository (with submodules)

```bash
git clone --recurse-submodules https://github.com/<your-username>/llm-tracer.git
cd llm-tracer
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Install build tools (one-time setup)

Install [MSYS2](https://www.msys2.org), then in the **MSYS2 UCRT64** terminal:

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
```

Add `C:\msys64\ucrt64\bin` to your Windows `PATH` environment variable, then restart your terminal.

### 3. Download a GGUF model

Download a GGUF model (e.g. [tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)) and place it in:

```
llm-tracer/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

> If you use a different model file name or path, update the `model_path` string in `src/tui/app.cpp`.

### 4. Build

```bash
mkdir build
cd build
cmake .. -G "Ninja" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
ninja
```

The first build compiles `llama.cpp`, `llama-common`, and `FTXUI` from source and may take several minutes.

### 5. Run

```bash
.\llm_tracer.exe
```

You'll be prompted for input text:

```
Enter a prompt to trace through the model: The capital of France is
```

The model loads (you'll see `llama.cpp`'s standard loading log), runs one forward pass over your prompt, and then launches into the TUI.

> **Note:** Run the executable from a standalone terminal window (Windows Terminal / cmd) rather than an editor's embedded terminal panel for reliable keyboard input, since some integrated terminals don't fully support the raw input mode FTXUI requires.

### Controls

| Key | Action |
|---|---|
| `↑` / `↓` or `j` / `k` | Navigate the layer list |
| `Tab` | Cycle focus between panels |
| `F` | Toggle attention matrix fullscreen |
| `Q` | Quit |

---

## Assumptions & Notes for Verification

- **CPU-only inference is assumed** by default in this build/config; no CUDA backend is configured. All "device" fields will read `CPU (Real Model)` — this is expected and is not flagged as an anomaly (the detector specifically distinguishes a genuine CPU-only run from a mid-run GPU→CPU fallback, which *would* be flagged as `CUDA OOM Fallback`).
- **Flash Attention is explicitly disabled** (`LLAMA_FLASH_ATTN_TYPE_DISABLED`) because `llama.cpp`'s fused flash-attention kernel never materializes a standalone `softmax(QK^T)` tensor — disabling it is required to expose the real attention matrix (`kq_soft_max`) for visualization. This is a deliberate trade-off: slightly slower inference in exchange for attention-weight observability.
- **Attention matrix viewport is fixed at 8×8** tokens (`ATTN_SIZE` in `ring_buffer.hpp`) and shows head 0 only, to keep the TUI panel a fixed, legible size regardless of prompt length or model head count. This can be changed by editing `ATTN_SIZE` and the head-indexing offset in `ggml_hooks.cpp`.
- **Latency anomaly threshold is adaptive, not fixed** — it is calibrated per-run as `mean + 2·stddev` across that run's own captured layers, since a fixed millisecond threshold would trigger constantly on CPU and almost never on GPU. This means a "quiet" run (no anomalies shown) does not indicate the detector is broken; it indicates no layer was a statistical outlier relative to its peers in that run.
- **One forward pass per prompt** — the current build runs a single `llama_decode` call over the full prompt (prefill), not multi-token autoregressive generation. This was a deliberate scope decision to keep the trace focused on per-layer compute behavior rather than token-by-token sampling.
- **Falls back to simulated data** if real model loading fails (missing GGUF file, bad path, etc.), so the TUI and panels remain demonstrable even without the model file present — useful for verifying UI behavior independently of inference correctness.
