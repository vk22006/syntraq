#pragma once

//
// syntraq/ai/traffic_predictor.hpp
//
// Native ONNX Runtime C++ inference engine for traffic prediction.
// Evaluates trained neural network models (Multi-Layer Perceptron)
// with sub-millisecond execution, asynchronous invocation, and telemetry.
//

#include "syntraq/ai/traffic_feature_scaler.hpp"
#include "syntraq/metrics/traffic_data_collector.hpp"

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace syntraq {

// Forward declare internal ONNX Runtime implementation details (PIMPL pattern)
// to keep ONNX headers out of consumer header include trees.
struct OrtSessionImpl;

class TrafficPredictor {
public:
    TrafficPredictor();
    ~TrafficPredictor();

    // Move-only semantics (PIMPL with unique_ptr)
    TrafficPredictor(const TrafficPredictor&) = delete;
    TrafficPredictor& operator=(const TrafficPredictor&) = delete;
    TrafficPredictor(TrafficPredictor&&) noexcept;
    TrafficPredictor& operator=(TrafficPredictor&&) noexcept;

    /// Load an ONNX model file from disk.
    /// Returns true on success, false on error without throwing exceptions.
    bool loadModel(const std::string& model_path);

    /// Predict directly from an already-preprocessed 26-dimensional float vector.
    /// Returns future predicted queue length (vehicles).
    [[nodiscard]] float predict(const std::vector<float>& preprocessed_features);

    /// Preprocess a raw simulation observation and evaluate inference.
    [[nodiscard]] float predictSample(
        const TrafficDataSample& sample,
        const std::string& scenario_name = ""
    );

    /// Batch inference across multiple preprocessed observations.
    [[nodiscard]] std::vector<float> predictBatch(
        const std::vector<std::vector<float>>& batch_features
    );

    /// Non-blocking asynchronous inference running on background worker.
    /// Prevents stalling render frames during simulation playback.
    [[nodiscard]] std::future<float> predictAsync(
        std::vector<float> preprocessed_features
    );

    /// Throttled control interval check (e.g. run inference every 0.5s or 1.0s).
    bool shouldRunInference(double current_sim_time, double interval_s = 0.5) noexcept;

    /// Query whether model is loaded and ready for execution.
    [[nodiscard]] bool isLoaded() const noexcept;

    /// Telemetry & performance metrics
    [[nodiscard]] float getLastInferenceLatencyUs() const noexcept;
    [[nodiscard]] float getAverageInferenceLatencyUs() const noexcept;
    [[nodiscard]] uint64_t getInferenceCount() const noexcept;
    void resetMetrics() noexcept;

    /// Retrieve descriptive error message if load or prediction failed.
    [[nodiscard]] std::string getLastError() const;

    /// Preprocessing scaler access
    [[nodiscard]] const TrafficFeatureScaler& getFeatureScaler() const noexcept;
    void setFeatureScaler(const TrafficFeatureScaler& scaler);

private:
    std::unique_ptr<OrtSessionImpl> impl_;
    TrafficFeatureScaler scaler_;
    bool is_loaded_             { false };
    std::string last_error_     {};

    // Telemetry
    float last_latency_us_      { 0.0f };
    double total_latency_us_    { 0.0 };
    uint64_t inference_count_   { 0 };
    double last_inference_time_ { -1.0 };
};

} // namespace syntraq
