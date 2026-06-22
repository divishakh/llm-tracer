#pragma once
#include "ring_buffer.hpp"
#include "detector.hpp"
#include <vector>
#include <ftxui/component/component.hpp>

class TracerApp {
public:
    TracerApp();
    void run();

private:
    RingBuffer<LayerEvent, 512> buffer;
    std::vector<Anomaly>        anomalies;
    std::vector<LayerEvent>     captured_events;
};