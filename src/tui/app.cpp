#include "tui/app.hpp"
#include "ggml_hooks.hpp"
#include "detector.hpp"
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <sstream>
#include <cmath>

using namespace ftxui;

TracerApp::TracerApp() {}

// Convert attention value to block character
std::string attn_to_block(float val) {
    if (val > 0.75f) return "██";
    if (val > 0.50f) return "▓▓";
    if (val > 0.25f) return "▒▒";
    if (val > 0.05f) return "░░";
    return "  ";
}

void TracerApp::run(const std::string& prompt) {
    const std::string model_path = "../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    bool ok = GGMLHooks::run_real_model(buffer, model_path, prompt);
    if (!ok) {
        GGMLHooks::simulate_forward_pass(buffer);
    }

    LayerEvent ev;
    while (buffer.pop(ev)) {
        captured_events.push_back(ev);
    }
    AnomalyDetector::calibrate(captured_events);
    for (auto& e : captured_events) {
        AnomalyDetector::check(e, anomalies);
    }

    auto screen = ScreenInteractive::Fullscreen();

    int selected_layer  = 0;
    int focused_panel   = 0;
    bool attn_fullscreen = false;
    int  attn_pan_x     = 0;
    int  attn_pan_y     = 0;

    std::vector<std::string> layer_names;
    for (auto& e : captured_events) {
        layer_names.push_back(
            "Layer " + std::to_string(e.layer_index)
            + " [" + e.layer_type + "]"
        );
    }

    auto topology = Menu(&layer_names, &selected_layer);

    auto renderer = Renderer(topology, [&] {
        LayerEvent sel = captured_events[selected_layer];

        // Helper: highlight focused panel with yellow border
        auto panel_color = [&](int panel_id, Element e) {
            if (focused_panel == panel_id)
                return e | borderStyled(ROUNDED) | color(Color::Yellow);
            return e;
        };

        // ── Panel 1: Model Topology ──
        auto topology_panel = panel_color(0,
            window(
                text(" 1. MODEL TOPOLOGY "),
                topology->Render()
            )
        );

        // ── Panel 2: Live Stream ──
        Elements rows;
        rows.push_back(hbox(
            text("ID   ") | bold,
            text("LAYER TYPE       ") | bold,
            text("DEVICE           ") | bold,
            text("LATENCY") | bold
        ));
        for (auto& e : captured_events) {
            rows.push_back(hbox(
                text(std::to_string(e.layer_index) + "    "),
                text(e.layer_type + "    "),
                text(e.device + "    "),
                text(std::to_string(e.latency_ms).substr(0,4) + " ms")
            ));
        }
        auto stream_panel = panel_color(1,
            window(text(" 2. LIVE PACKET STREAM "), vbox(rows))
        );

        // ── Panel 3: Attention Matrix ──
        Elements attn_rows;

        // Token header row
        Elements header;
        header.push_back(text("      "));
        for (int c = 0; c < sel.num_tokens; c++) {
            std::string tok = sel.tokens[c];
            if (tok.size() > 6) tok = tok.substr(0, 6);
            while (tok.size() < 6) tok += " ";
            header.push_back(text(tok));
        }
        attn_rows.push_back(hbox(header));
        attn_rows.push_back(separator());

        // Matrix rows
        for (int row = 0; row < sel.num_tokens; row++) {
            Elements row_elems;

            // Row label
            std::string label = "[" + sel.tokens[row] + "]";
            while (label.size() < 6) label += " ";
            row_elems.push_back(text(label));

            for (int col = 0; col < sel.num_tokens; col++) {
                float val = sel.attn_matrix[row][col];
                std::string block = attn_to_block(val);

                // Color the block based on intensity
                Element e = text(block);
                if (val > 0.75f)      e = e | color(Color::Red);
                else if (val > 0.50f) e = e | color(Color::Yellow);
                else if (val > 0.25f) e = e | color(Color::Green);
                else                  e = e | color(Color::GrayDark);

                row_elems.push_back(e);
            }
            attn_rows.push_back(hbox(row_elems));
        }

        // Controls hint
        attn_rows.push_back(separator());
        attn_rows.push_back(
            text("[F]: Fullscreen  [Arrows]: Pan  [+/-]: Contrast") | dim
        );

        auto attn_panel = panel_color(2,
            window(
                text(" 3. ATTENTION MATRIX (HEAD 0) "),
                vbox(attn_rows)
            )
        );

        // ── Panel 4: Runtime Metrics ──
        std::string shape = "[" + std::to_string(sel.shape[0])
                          + ", " + std::to_string(sel.shape[1])
                          + ", " + std::to_string(sel.shape[2]) + "]";

        int bar = (int)(sel.sparsity / 10);
        std::string sparsity_bar;
        for (int i = 0; i < 10; i++)
            sparsity_bar += (i < bar) ? "█" : "░";

        auto metrics_panel = panel_color(3,
            window(text(" 4. RUNTIME METRICS "),
                vbox({
                    text("Tensor Shape : " + shape),
                    text("Sparsity     : " + sparsity_bar
                         + " " + std::to_string((int)sel.sparsity) + "%"),
                    text("Latency      : "
                         + std::to_string(sel.latency_ms).substr(0,4) + " ms"),
                    text("Mean         : "
                         + std::to_string(sel.mean).substr(0,4)),
                    text("Max Value    : "
                         + std::to_string(sel.max_val).substr(0,4)),
                    text("Device       : " + sel.device),
                })
            )
        );

        // ── Panel 5: Anomaly Ledger ──
        Elements anomaly_rows;
        if (anomalies.empty()) {
            anomaly_rows.push_back(text("No anomalies detected"));
        } else {
            for (auto& a : anomalies) {
                std::string prefix = (a.severity == "ERROR") ? "ERROR: " : "WARN: ";
                anomaly_rows.push_back(
                    text(prefix + a.message) | color(
                        a.severity == "ERROR" ? Color::Red : Color::Yellow
                    )
                );
            }
        }
        auto anomaly_panel = panel_color(4,
            window(text(" 5. ANOMALY LEDGER "), vbox(anomaly_rows))
        );

        // Status bar
        std::string focus_names[] = {
            "TOPOLOGY", "STREAM", "ATTENTION MATRIX", "METRICS", "ANOMALIES"
        };
        std::string status = "Focus: [" + focus_names[focused_panel] + "]";

        // If fullscreen attention matrix
        if (attn_fullscreen) {
            return vbox({
                hbox(
                    text("LLM Tracer - ATTENTION FULLSCREEN") | bold,
                    filler(),
                    text("[F]: Exit Fullscreen  [Q]: Quit") | color(Color::Yellow)
                ),
                separator(),
                attn_panel | flex
            });
        }

        // Normal layout
        return vbox({
            hbox(
                text("LLM Tracer") | bold,
                text("  |  [Tab]: Focus  |  [j/k]: Navigate  |  [F]: Attn Fullscreen  |  [Q]: Quit"),
                filler(),
                text(status) | color(Color::Yellow) | bold
            ),
            separator(),
            hbox(topology_panel | xflex, stream_panel | xflex),
            hbox(attn_panel | xflex, metrics_panel | xflex),
            anomaly_panel
        });
    });

    // Keyboard handler
    auto component = CatchEvent(renderer, [&](Event event) {
        if (event == Event::Tab) {
            focused_panel = (focused_panel + 1) % 5;
            return true;
        }
        if (event == Event::Character('f') || event == Event::Character('F')) {
            attn_fullscreen = !attn_fullscreen;
            return true;
        }
        if (event == Event::Character('q') || event == Event::Character('Q')) {
            screen.ExitLoopClosure()();
            return true;
        }
        // j/k as vim-style aliases for the layer menu
        if (event == Event::Character('j')) {
            if (selected_layer < (int)captured_events.size() - 1) selected_layer++;
            return true;
        }
        if (event == Event::Character('k')) {
            if (selected_layer > 0) selected_layer--;
            return true;
        }
        return false;
    });

    screen.Loop(component);
}