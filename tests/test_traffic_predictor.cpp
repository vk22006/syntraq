//
// tests/test_traffic_predictor.cpp
//
// Unit tests for TrafficPredictor native ONNX Runtime inference engine:
// - Model loading and graceful error handling
// - Feature preprocessing parity (continuous scaling and one-hot encoding)
// - Synchronous and asynchronous inference execution
// - Inference latency tracking and performance telemetry
// - Automated parity validation against Python ONNX Runtime reference predictions
//

#include <gtest/gtest.h>

#include "syntraq/ai/traffic_feature_scaler.hpp"
#include "syntraq/ai/traffic_predictor.hpp"
#include "syntraq/metrics/traffic_data_collector.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <vector>

namespace syntraq {
namespace {

std::string resolvePath(const std::string& rel_path) {
    if (std::filesystem::exists(rel_path)) {
        return rel_path;
    }
    if (std::filesystem::exists("../" + rel_path)) {
        return "../" + rel_path;
    }
    if (std::filesystem::exists("../../" + rel_path)) {
        return "../../" + rel_path;
    }
    return rel_path;
}

const std::string kValidModelPath = resolvePath("models/traffic_predictor.onnx");
const std::string kValidationJsonPath = resolvePath("data/dataset/inference_validation_samples.json");

// ── Test 1: Model Loading & Error Handling ───────────────────────────────────

TEST(TrafficPredictorTest, PredictorStartsUnloadedAndHandlesInvalidPath) {
    TrafficPredictor predictor;
    EXPECT_FALSE(predictor.isLoaded());

    // Predict on unloaded predictor must fail gracefully without throwing
    std::vector<float> dummy(26, 0.0f);
    float result = predictor.predict(dummy);
    EXPECT_FLOAT_EQ(result, 0.0f);
    EXPECT_FALSE(predictor.getLastError().empty());

    // Attempt loading non-existent model
    bool loaded = predictor.loadModel("non_existent_dir/missing_model.onnx");
    EXPECT_FALSE(loaded);
    EXPECT_FALSE(predictor.isLoaded());
    EXPECT_NE(predictor.getLastError().find("does not exist"), std::string::npos);
}

TEST(TrafficPredictorTest, PredictorLoadsValidOnnxModel) {
    if (!std::filesystem::exists(kValidModelPath)) {
        GTEST_SKIP() << "ONNX model not found at: " << kValidModelPath;
    }

    TrafficPredictor predictor;
    bool loaded = predictor.loadModel(kValidModelPath);
    EXPECT_TRUE(loaded);
    EXPECT_TRUE(predictor.isLoaded());
    EXPECT_TRUE(predictor.getLastError().empty());
}

// ── Test 2: Preprocessing Scaling & Categorical One-Hot ───────────────────────

TEST(TrafficPredictorTest, FeatureScalerNormalizesSampleAccurately) {
    TrafficFeatureScaler scaler;

    // Zero input test
    const auto& vc_scaler = scaler.getScaler("vehicle_count");
    float expected_scaled_zero = (0.0f - vc_scaler.mean) / vc_scaler.std;
    EXPECT_NEAR(vc_scaler.scale(0.0f), expected_scaled_zero, 1e-5f);

    // Mean input test (should map to 0.0)
    EXPECT_NEAR(vc_scaler.scale(vc_scaler.mean), 0.0f, 1e-5f);

    // TrafficDataSample transform
    TrafficDataSample sample;
    sample.vehicle_count = 30;
    sample.queue_vehicle_count = 5;
    sample.scenario_name = "Medium Traffic";

    std::vector<float> transformed = scaler.transform(sample);
    ASSERT_EQ(transformed.size(), TrafficFeatureScaler::TOTAL_FEATURES);

    // Feature 0: vehicle_count
    float expected_vc = (30.0f - vc_scaler.mean) / vc_scaler.std;
    EXPECT_NEAR(transformed[0], expected_vc, 1e-4f);

    // Scenario one-hot flags at indices [22, 23, 24, 25]:
    // Low, Med, High, Rush
    EXPECT_FLOAT_EQ(transformed[22], 0.0f); // Low
    EXPECT_FLOAT_EQ(transformed[23], 1.0f); // Med
    EXPECT_FLOAT_EQ(transformed[24], 0.0f); // High
    EXPECT_FLOAT_EQ(transformed[25], 0.0f); // Rush
}

TEST(TrafficPredictorTest, FeatureScalerOneHotEncodesAllScenarios) {
    TrafficFeatureScaler scaler;
    TrafficDataSample sample;

    // Test Low
    auto low_vec = scaler.transform(sample, "Low Traffic");
    EXPECT_FLOAT_EQ(low_vec[22], 1.0f);
    EXPECT_FLOAT_EQ(low_vec[23], 0.0f);

    // Test High
    auto high_vec = scaler.transform(sample, "High Traffic");
    EXPECT_FLOAT_EQ(high_vec[24], 1.0f);

    // Test Rush Hour
    auto rush_vec = scaler.transform(sample, "Rush Hour");
    EXPECT_FLOAT_EQ(rush_vec[25], 1.0f);
}

// ── Test 3: Prediction Execution & Parity ─────────────────────────────────────

TEST(TrafficPredictorTest, PredictSampleMatchesDirectPredict) {
    if (!std::filesystem::exists(kValidModelPath)) {
        GTEST_SKIP() << "ONNX model not found at: " << kValidModelPath;
    }

    TrafficPredictor predictor;
    ASSERT_TRUE(predictor.loadModel(kValidModelPath));

    TrafficDataSample sample;
    sample.vehicle_count = 45;
    sample.spawn_rate_vpm = 80.0f;
    sample.queue_vehicle_count = 12;
    sample.max_queue_length = 3;
    sample.avg_queue_per_lane = 0.5f;
    sample.queue_ratio = 0.27f;
    sample.avg_speed_mps = 6.5f;
    sample.avg_speed_kmh = 23.4f;
    sample.road_occupancy_pct = 4.2f;
    sample.scenario_name = "Rush Hour";

    // 1. Direct prediction on preprocessed vector
    std::vector<float> preprocessed = predictor.getFeatureScaler().transform(sample);
    float direct_pred = predictor.predict(preprocessed);

    // 2. Convenience predictSample()
    float sample_pred = predictor.predictSample(sample);

    EXPECT_NEAR(direct_pred, sample_pred, 1e-5f);
    EXPECT_FALSE(std::isnan(direct_pred));
    EXPECT_FALSE(std::isinf(direct_pred));
}

// ── Test 4: Latency & Telemetry Metrics ───────────────────────────────────────

TEST(TrafficPredictorTest, InferenceLatencyTelemetryIsCaptured) {
    if (!std::filesystem::exists(kValidModelPath)) {
        GTEST_SKIP() << "ONNX model not found at: " << kValidModelPath;
    }

    TrafficPredictor predictor;
    ASSERT_TRUE(predictor.loadModel(kValidModelPath));
    predictor.resetMetrics();
    EXPECT_EQ(predictor.getInferenceCount(), 0u);
    EXPECT_FLOAT_EQ(predictor.getLastInferenceLatencyUs(), 0.0f);

    std::vector<float> dummy(26, 0.0f);
    for (int i = 0; i < 10; ++i) {
        float p = predictor.predict(dummy);
        (void)p;
    }

    EXPECT_EQ(predictor.getInferenceCount(), 10u);
    EXPECT_GT(predictor.getLastInferenceLatencyUs(), 0.0f);
    EXPECT_GT(predictor.getAverageInferenceLatencyUs(), 0.0f);
}

// ── Test 5: Asynchronous Inference Execution ─────────────────────────────────

TEST(TrafficPredictorTest, AsynchronousInferenceCompletesSuccessfully) {
    if (!std::filesystem::exists(kValidModelPath)) {
        GTEST_SKIP() << "ONNX model not found at: " << kValidModelPath;
    }

    TrafficPredictor predictor;
    ASSERT_TRUE(predictor.loadModel(kValidModelPath));

    std::vector<float> dummy(26, 0.0f);
    std::future<float> future_pred = predictor.predictAsync(dummy);

    // Simulate work during non-blocking inference
    std::future_status status = future_pred.wait_for(std::chrono::milliseconds(100));
    EXPECT_EQ(status, std::future_status::ready);

    float async_result = future_pred.get();
    float sync_result = predictor.predict(dummy);

    EXPECT_NEAR(async_result, sync_result, 1e-4f);
}

// ── Test 6: Automated Validation Against Python Reference Predictions ────────

TEST(TrafficPredictorTest, AutomatedPythonParityValidationTest) {
    if (!std::filesystem::exists(kValidModelPath)) {
        GTEST_SKIP() << "ONNX model not found at: " << kValidModelPath;
    }
    if (!std::filesystem::exists(kValidationJsonPath)) {
        GTEST_SKIP() << "Validation reference samples not found at: " << kValidationJsonPath;
    }

    TrafficPredictor predictor;
    ASSERT_TRUE(predictor.loadModel(kValidModelPath));

    // Parse validation JSON
    std::ifstream file(kValidationJsonPath);
    ASSERT_TRUE(file.is_open());
    std::string json_content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());

    // Verify sample entries exist
    std::size_t sample_pos = json_content.find("\"samples\":");
    ASSERT_NE(sample_pos, std::string::npos);

    // Simple robust parser to extract each sample's features and expected_prediction
    std::size_t curr = sample_pos;
    int samples_tested = 0;

    while (true) {
        auto feat_pos = json_content.find("\"features\":", curr);
        if (feat_pos == std::string::npos) break;

        auto bracket_open = json_content.find("[", feat_pos);
        auto bracket_close = json_content.find("]", bracket_open);
        if (bracket_open == std::string::npos || bracket_close == std::string::npos) break;

        std::string feats_str = json_content.substr(bracket_open + 1, bracket_close - bracket_open - 1);
        std::vector<float> features;
        std::stringstream ss(feats_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if (!token.empty()) {
                features.push_back(std::stof(token));
            }
        }

        auto pred_pos = json_content.find("\"expected_prediction\":", bracket_close);
        ASSERT_NE(pred_pos, std::string::npos);
        auto val_start = json_content.find_first_of("0123456789.-", pred_pos + 22);
        auto val_end = json_content.find_first_of(",\n\r}", val_start);
        float expected_prediction = std::stof(json_content.substr(val_start, val_end - val_start));

        ASSERT_EQ(features.size(), 26u);

        // Run C++ ONNX Runtime prediction
        float cpp_pred = predictor.predict(features);

        // Assert numerical match within tight tolerance
        float diff = std::abs(cpp_pred - expected_prediction);
        EXPECT_LT(diff, 1e-3f)
            << "Sample " << samples_tested
            << " discrepancy: C++=" << cpp_pred
            << " vs Python=" << expected_prediction;

        samples_tested++;
        curr = val_end;
    }

    EXPECT_GT(samples_tested, 0);
    std::cout << "[AutomatedParityTest] Successfully verified "
              << samples_tested << " reference samples against Python predictions with discrepancy < 1e-3."
              << std::endl;
}

} // namespace
} // namespace syntraq
