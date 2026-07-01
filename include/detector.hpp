#pragma once
#include "ring_buffer.hpp"
#include <string>
#include <vector>

struct Anomaly {
    std::string message;
    std::string severity;
    long long   timestamp;
};

class AnomalyDetector {
public:
    static void check(const LayerEvent& ev, std::vector<Anomaly>& anomalies);

    // Call this once after all events are collected to compute
    // a realistic latency baseline before running check() again
    static void calibrate(const std::vector<LayerEvent>& events);

private:
    static constexpr float MAX_ACTIVATION_THRESHOLD = 6.0f;

    static float latency_mean;
    static float latency_stddev;
    static bool  calibrated;
};