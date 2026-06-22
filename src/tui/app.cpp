#include "tui/app.hpp"
#include "ggml_hooks.hpp"
#include "detector.hpp"
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <sstream>

using namespace ftxui;

TracerApp::TracerApp() {}

void TracerApp::run() {
    GGMLHooks::simulate_forward_pass(buffer);

    LayerEvent ev;
    while (buffer.pop(ev)) {
        AnomalyDetector::check(ev, anomalies);
        captured_events.push_back(ev);
    }

    auto screen = ScreenInteractive::Fullscreen();

    int selected_layer = 0;
    std::vector<std::string> layer_names;
    for (auto& e : captured_events) {
        layer_names.push_back(
            "Layer " + std::to_string(e.layer_index) + " [" + e.layer_type + "]"
        );
    }

    auto topology = Menu(&layer_names, &selected_layer);

    auto renderer = Renderer(topology, [&] {
        LayerEvent sel = captured_events[selected_layer];

        // Panel 1 - Topology
        auto topology_panel = window(
            text(" 1. MODEL TOPOLOGY "),
            topology->Render()
        );

        // Panel 2 - Stream
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
        auto stream_panel = window(
            text(" 2. LIVE STREAM "),
            vbox(rows)
        );

        // Panel 3 - Metrics
        std::string shape = "[" + std::to_string(sel.shape[0])
                          + ", " + std::to_string(sel.shape[1])
                          + ", " + std::to_string(sel.shape[2]) + "]";

        int bar = (int)(sel.sparsity / 10);
        std::string sparsity_bar;
        for (int i = 0; i < 10; i++)
            sparsity_bar += (i < bar) ? "█" : "░";

        auto metrics_panel = window(
            text(" 4. RUNTIME METRICS "),
            vbox({
                text("Tensor Shape : " + shape),
                text("Sparsity     : " + sparsity_bar + " " + std::to_string((int)sel.sparsity) + "%"),
                text("Latency      : " + std::to_string(sel.latency_ms).substr(0,4) + " ms"),
                text("Mean         : " + std::to_string(sel.mean).substr(0,4)),
                text("Max Value    : " + std::to_string(sel.max_val).substr(0,4)),
                text("Device       : " + sel.device),
            })
        );

        // Panel 4 - Anomalies
        Elements anomaly_rows;
        if (anomalies.empty()) {
            anomaly_rows.push_back(text("No anomalies detected"));
        } else {
            for (auto& a : anomalies) {
                std::string prefix = (a.severity == "ERROR") ? "ERROR: " : "WARN: ";
                anomaly_rows.push_back(text(prefix + a.message));
            }
        }
        auto anomaly_panel = window(
            text(" 5. ANOMALY LEDGER "),
            vbox(anomaly_rows)
        );

        return vbox({
            text("LLM Tracer  |  [Tab]: Cycle Focus  |  [Q]: Quit") | bold | center,
            separator(),
            hbox(topology_panel | xflex, stream_panel | xflex),
            hbox(metrics_panel  | xflex, anomaly_panel | xflex)
        });
    });

    auto component = CatchEvent(renderer, [&](Event event) {
        if (event == Event::Character('q') || event == Event::Character('Q')) {
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    screen.Loop(component);
}