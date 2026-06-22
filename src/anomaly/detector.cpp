#include "detector.hpp"
#include <sstream>

void AnomalyDetector::check(const LayerEvent& ev, std::vector<Anomaly>& anomalies) {
    Anomaly a;
    a.timestamp = ev.timestamp;

    // Check if max activation is too high
    if (ev.max_val > MAX_ACTIVATION_THRESHOLD) {
        std::ostringstream msg;
        msg << "Outlier Feature Layer " << ev.layer_index
            << ": Max > " << MAX_ACTIVATION_THRESHOLD;
        a.message  = msg.str();
        a.severity = "WARNING";
        anomalies.push_back(a);
    }

    // Check if layer is running on CPU instead of CUDA
    if (ev.device.find("CPU") != std::string::npos) {
        std::ostringstream msg;
        msg << "CUDA OOM Fallback: Layer " << ev.layer_index
            << " running on CPU Host Memory";
        a.message  = msg.str();
        a.severity = "ERROR";
        anomalies.push_back(a);
    }

    // Check if latency is too high
    if (ev.latency_ms > HIGH_LATENCY_THRESHOLD) {
        std::ostringstream msg;
        msg << "High Latency on Layer " << ev.layer_index
            << ": " << ev.latency_ms << " ms";
        a.message  = msg.str();
        a.severity = "WARNING";
        anomalies.push_back(a);
    }
}