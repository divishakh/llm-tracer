#pragma once
#include "ring_buffer.hpp"
#include "detector.hpp"
#include <vector>
#include <string>
#include <ftxui/component/component.hpp>

class TracerApp {
public:
    TracerApp();
    void run(const std::string& prompt);

private:
    RingBuffer<LayerEvent, 512> buffer;
    std::vector<Anomaly>        anomalies;
    std::vector<LayerEvent>     captured_events;
};