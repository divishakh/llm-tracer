#include "detector.hpp"
#include <sstream>
#include <cmath>

float AnomalyDetector::latency_mean   = 0.0f;
float AnomalyDetector::latency_stddev = 0.0f;
bool  AnomalyDetector::calibrated     = false;

void AnomalyDetector::calibrate(const std::vector<LayerEvent>& events) {
    if (events.empty()) return;

    double sum = 0.0;
    for (auto& e : events) sum += e.latency_ms;
    float mean = (float)(sum / events.size());

    double sq_sum = 0.0;
    for (auto& e : events) {
        float diff = e.latency_ms - mean;
        sq_sum += diff * diff;
    }
    float stddev = (float)sqrt(sq_sum / events.size());

    latency_mean   = mean;
    latency_stddev = stddev;
    calibrated     = true;
}

void AnomalyDetector::check(const LayerEvent& ev, std::vector<Anomaly>& anomalies) {
    Anomaly a;
    a.timestamp = ev.timestamp;

    // Outlier activation check (unchanged)
    if (ev.max_val > MAX_ACTIVATION_THRESHOLD) {
        std::ostringstream msg;
        msg << "Outlier Feature Layer " << ev.layer_index
            << ": Max > " << MAX_ACTIVATION_THRESHOLD;
        a.message  = msg.str();
        a.severity = "WARNING";
        anomalies.push_back(a);
    }

    // CPU fallback check (only meaningful if model is supposed to be on GPU)
    if (ev.device.find("CPU (Fallback)") != std::string::npos) {
        std::ostringstream msg;
        msg << "CUDA OOM Fallback: Layer " << ev.layer_index
            << " running on CPU Host Memory";
        a.message  = msg.str();
        a.severity = "ERROR";
        anomalies.push_back(a);
    }

    // Adaptive latency check - only flags layers that are real outliers
    // relative to this run's own average, not a fixed number
    if (calibrated && latency_stddev > 0.0f) {
        float z = (ev.latency_ms - latency_mean) / latency_stddev;
        if (z > 2.0f) {
            std::ostringstream msg;
            msg << "High Latency on Layer " << ev.layer_index
                << ": " << ev.latency_ms << " ms (avg "
                << latency_mean << " ms)";
            a.message  = msg.str();
            a.severity = "WARNING";
            anomalies.push_back(a);
        }
    }
}