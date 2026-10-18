//
// syntraq/ai/traffic_predictor.cpp
//
// Implementation of native ONNX Runtime C++ inference engine.
//

#include "syntraq/ai/traffic_predictor.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

namespace syntraq {

struct OrtSessionImpl {
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "SyntraQ_TrafficPredictor" };
    Ort::SessionOptions session_options{};
    std::unique_ptr<Ort::Session> session{};
    Ort::MemoryInfo memory_info{ Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) };
    std::string input_name{ "input" };
    std::string output_name{ "output" };
};

TrafficPredictor::TrafficPredictor()
    : impl_(std::make_unique<OrtSessionImpl>()) {
}

TrafficPredictor::~TrafficPredictor() = default;

TrafficPredictor::TrafficPredictor(TrafficPredictor&&) noexcept = default;
TrafficPredictor& TrafficPredictor::operator=(TrafficPredictor&&) noexcept = default;

bool TrafficPredictor::loadModel(const std::string& model_path) {
    if (!std::filesystem::exists(model_path)) {
        last_error_ = "Model file does not exist: " + model_path;
        is_loaded_ = false;
        return false;
    }

    try {
        impl_->session_options.SetIntraOpNumThreads(1);
        impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring wide_path(model_path.begin(), model_path.end());
        impl_->session = std::make_unique<Ort::Session>(impl_->env, wide_path.c_str(), impl_->session_options);
#else
        impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), impl_->session_options);
#endif

        // Inspect input name
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = impl_->session->GetInputNameAllocated(0, allocator);
        if (input_name_ptr) {
            impl_->input_name = input_name_ptr.get();
        }

        // Inspect output name
        auto output_name_ptr = impl_->session->GetOutputNameAllocated(0, allocator);
        if (output_name_ptr) {
            impl_->output_name = output_name_ptr.get();
        }

        is_loaded_ = true;
        last_error_.clear();
        return true;
    } catch (const Ort::Exception& ex) {
        last_error_ = std::string("ONNX Runtime Exception: ") + ex.what();
        is_loaded_ = false;
        return false;
    } catch (const std::exception& ex) {
        last_error_ = std::string("Model load error: ") + ex.what();
        is_loaded_ = false;
        return false;
    } catch (...) {
        last_error_ = "Unknown error occurred while loading ONNX model.";
        is_loaded_ = false;
        return false;
    }
}

bool TrafficPredictor::isLoaded() const noexcept {
    return is_loaded_ && impl_ && impl_->session != nullptr;
}

float TrafficPredictor::predict(const std::vector<float>& preprocessed_features) {
    if (!isLoaded()) {
        last_error_ = "predict() called on uninitialized or unloaded model.";
        return 0.0f;
    }

    if (preprocessed_features.size() != TrafficFeatureScaler::TOTAL_FEATURES) {
        last_error_ = "Feature dimension mismatch: expected " +
                      std::to_string(TrafficFeatureScaler::TOTAL_FEATURES) +
                      ", got " + std::to_string(preprocessed_features.size());
        return 0.0f;
    }

    const auto t_start = std::chrono::high_resolution_clock::now();

    try {
        std::array<int64_t, 2> input_shape{ 1, static_cast<int64_t>(preprocessed_features.size()) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            impl_->memory_info,
            const_cast<float*>(preprocessed_features.data()),
            preprocessed_features.size(),
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = { impl_->input_name.c_str() };
        const char* output_names[] = { impl_->output_name.c_str() };

        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{ nullptr },
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        const float* out_data = output_tensors[0].GetTensorData<float>();
        const float prediction = out_data[0];

        const auto t_end = std::chrono::high_resolution_clock::now();
        const float elapsed_us = std::chrono::duration<float, std::micro>(t_end - t_start).count();

        last_latency_us_ = elapsed_us;
        total_latency_us_ += elapsed_us;
        inference_count_++;

        return prediction;
    } catch (const Ort::Exception& ex) {
        last_error_ = std::string("ONNX Inference Exception: ") + ex.what();
        return 0.0f;
    } catch (const std::exception& ex) {
        last_error_ = std::string("Inference Exception: ") + ex.what();
        return 0.0f;
    }
}

float TrafficPredictor::predictSample(
    const TrafficDataSample& sample,
    const std::string& scenario_name
) {
    const std::vector<float> preprocessed = scaler_.transform(sample, scenario_name);
    return predict(preprocessed);
}

std::vector<float> TrafficPredictor::predictBatch(
    const std::vector<std::vector<float>>& batch_features
) {
    std::vector<float> predictions;
    if (batch_features.empty()) return predictions;

    if (!isLoaded()) {
        last_error_ = "predictBatch() called on uninitialized model.";
        predictions.resize(batch_features.size(), 0.0f);
        return predictions;
    }

    const std::size_t batch_size = batch_features.size();
    predictions.reserve(batch_size);

    // Flatten into contiguous buffer
    std::vector<float> flat_buffer;
    flat_buffer.reserve(batch_size * TrafficFeatureScaler::TOTAL_FEATURES);
    for (const auto& row : batch_features) {
        if (row.size() != TrafficFeatureScaler::TOTAL_FEATURES) {
            last_error_ = "Row in batch features has invalid dimension: " + std::to_string(row.size());
            predictions.resize(batch_size, 0.0f);
            return predictions;
        }
        flat_buffer.insert(flat_buffer.end(), row.begin(), row.end());
    }

    const auto t_start = std::chrono::high_resolution_clock::now();

    try {
        std::array<int64_t, 2> input_shape{
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(TrafficFeatureScaler::TOTAL_FEATURES)
        };

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            impl_->memory_info,
            flat_buffer.data(),
            flat_buffer.size(),
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = { impl_->input_name.c_str() };
        const char* output_names[] = { impl_->output_name.c_str() };

        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{ nullptr },
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        const float* out_data = output_tensors[0].GetTensorData<float>();
        for (std::size_t i = 0; i < batch_size; ++i) {
            predictions.push_back(out_data[i]);
        }

        const auto t_end = std::chrono::high_resolution_clock::now();
        const float elapsed_us = std::chrono::duration<float, std::micro>(t_end - t_start).count();

        last_latency_us_ = elapsed_us / static_cast<float>(batch_size);
        total_latency_us_ += elapsed_us;
        inference_count_ += batch_size;

        return predictions;
    } catch (const Ort::Exception& ex) {
        last_error_ = std::string("Batch Inference Exception: ") + ex.what();
        predictions.resize(batch_size, 0.0f);
        return predictions;
    }
}

std::future<float> TrafficPredictor::predictAsync(
    std::vector<float> preprocessed_features
) {
    return std::async(
        std::launch::async,
        [this, feats = std::move(preprocessed_features)]() {
            return this->predict(feats);
        }
    );
}

bool TrafficPredictor::shouldRunInference(
    double current_sim_time,
    double interval_s
) noexcept {
    if (last_inference_time_ < 0.0 || (current_sim_time - last_inference_time_) >= interval_s) {
        last_inference_time_ = current_sim_time;
        return true;
    }
    return false;
}

float TrafficPredictor::getLastInferenceLatencyUs() const noexcept {
    return last_latency_us_;
}

float TrafficPredictor::getAverageInferenceLatencyUs() const noexcept {
    if (inference_count_ == 0) return 0.0f;
    return static_cast<float>(total_latency_us_ / static_cast<double>(inference_count_));
}

uint64_t TrafficPredictor::getInferenceCount() const noexcept {
    return inference_count_;
}

void TrafficPredictor::resetMetrics() noexcept {
    last_latency_us_ = 0.0f;
    total_latency_us_ = 0.0;
    inference_count_ = 0;
    last_inference_time_ = -1.0;
}

std::string TrafficPredictor::getLastError() const {
    return last_error_;
}

const TrafficFeatureScaler& TrafficPredictor::getFeatureScaler() const noexcept {
    return scaler_;
}

void TrafficPredictor::setFeatureScaler(const TrafficFeatureScaler& scaler) {
    scaler_ = scaler;
}

} // namespace syntraq
