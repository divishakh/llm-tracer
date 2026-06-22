#pragma once
#include "ring_buffer.hpp"
#include <string>
#include <vector>

// Represents a detected anomaly
struct Anomaly {
    std::string message;    // description of the anomaly
    std::string severity;   // "WARNING" or "ERROR"
    long long   timestamp;
};

// Checks each layer event for numerical anomalies
class AnomalyDetector {
public:
    // Check a single layer event for anomalies
    static void check(const LayerEvent& ev, std::vector<Anomaly>& anomalies);

private:
    static constexpr float MAX_ACTIVATION_THRESHOLD = 6.0f;
    static constexpr float HIGH_LATENCY_THRESHOLD   = 1.5f;
};