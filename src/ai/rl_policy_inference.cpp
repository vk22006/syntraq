//
// syntraq/ai/rl_policy_inference.cpp
//
// Implementation of native ONNX Runtime C++ inference engine for trained DQN policies.
//

#include "syntraq/ai/rl_policy_inference.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <string>

namespace syntraq {

struct OrtPolicySessionImpl {
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "SyntraQ_RLPolicyInference" };
    Ort::SessionOptions session_options{};
    std::unique_ptr<Ort::Session> session{};
    Ort::MemoryInfo memory_info{ Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) };
    Ort::RunOptions run_options{ nullptr };
    std::string input_name{ "observation" };
    std::string output_name{ "q_values" };
    std::vector<float> flat_input_buffer{};
    std::vector<float> flat_output_buffer{};
};

RLPolicyInference::RLPolicyInference()
    : impl_(std::make_unique<OrtPolicySessionImpl>()) {
}

RLPolicyInference::~RLPolicyInference() = default;

RLPolicyInference::RLPolicyInference(RLPolicyInference&&) noexcept = default;
RLPolicyInference& RLPolicyInference::operator=(RLPolicyInference&&) noexcept = default;

bool RLPolicyInference::loadModel(const std::string& model_path) {
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

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = impl_->session->GetInputNameAllocated(0, allocator);
        if (input_name_ptr) {
            impl_->input_name = input_name_ptr.get();
        }

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
        last_error_ = "Unknown error occurred while loading ONNX RL policy model.";
        is_loaded_ = false;
        return false;
    }
}

bool RLPolicyInference::isLoaded() const noexcept {
    return is_loaded_ && impl_ && impl_->session != nullptr;
}

RLActionDecision RLPolicyInference::predict(const std::vector<float>& observation) {
    if (!isLoaded()) {
        last_error_ = "predict() called on uninitialized or unloaded model.";
        return RLActionDecision{ 0, 0.0f, 0.0f, 0.0f };
    }

    if (observation.size() != OBSERVATION_DIM) {
        last_error_ = "Observation dimension mismatch: expected " +
                      std::to_string(OBSERVATION_DIM) +
                      ", got " + std::to_string(observation.size());
        return RLActionDecision{ 0, 0.0f, 0.0f, 0.0f };
    }

    const auto t_start = std::chrono::high_resolution_clock::now();

    try {
        std::array<int64_t, 2> input_shape{ 1, static_cast<int64_t>(OBSERVATION_DIM) };
        std::array<int64_t, 2> output_shape{ 1, static_cast<int64_t>(ACTION_DIM) };
        std::array<float, ACTION_DIM> output_buf{ 0.0f, 0.0f };

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            impl_->memory_info,
            const_cast<float*>(observation.data()),
            observation.size(),
            input_shape.data(),
            input_shape.size()
        );
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            impl_->memory_info,
            output_buf.data(),
            output_buf.size(),
            output_shape.data(),
            output_shape.size()
        );

        const char* input_names[] = { impl_->input_name.c_str() };
        const char* output_names[] = { impl_->output_name.c_str() };

        impl_->session->Run(
            impl_->run_options,
            input_names,
            &input_tensor,
            1,
            output_names,
            &output_tensor,
            1
        );

        const float q0 = output_buf[0];
        const float q1 = output_buf[1];
        const int best_action = (q1 > q0) ? 1 : 0;

        const auto t_end = std::chrono::high_resolution_clock::now();
        const float latency_us = static_cast<float>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()
        ) / 1000.0f;

        last_latency_us_ = latency_us;
        total_latency_us_ += latency_us;
        inference_count_++;
        last_error_.clear();

        return RLActionDecision{ best_action, q0, q1, latency_us };
    } catch (const Ort::Exception& ex) {
        last_error_ = std::string("ONNX Runtime Evaluation Exception: ") + ex.what();
        return RLActionDecision{ 0, 0.0f, 0.0f, 0.0f };
    } catch (const std::exception& ex) {
        last_error_ = std::string("Inference exception: ") + ex.what();
        return RLActionDecision{ 0, 0.0f, 0.0f, 0.0f };
    }
}

std::vector<RLActionDecision> RLPolicyInference::predictBatchFlat(
    const float* flat_observations,
    size_t       batch_size
) {
    std::vector<RLActionDecision> decisions;
    if (batch_size == 0 || !flat_observations) return decisions;

    if (!isLoaded()) {
        last_error_ = "predictBatchFlat() called on uninitialized or unloaded model.";
        decisions.resize(batch_size, RLActionDecision{ 0, 0.0f, 0.0f, 0.0f });
        return decisions;
    }

    const auto t_start = std::chrono::high_resolution_clock::now();

    try {
        impl_->flat_output_buffer.resize(batch_size * ACTION_DIM);
        std::array<int64_t, 2> input_shape{
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(OBSERVATION_DIM)
        };
        std::array<int64_t, 2> output_shape{
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(ACTION_DIM)
        };

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            impl_->memory_info,
            const_cast<float*>(flat_observations),
            batch_size * OBSERVATION_DIM,
            input_shape.data(),
            input_shape.size()
        );
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            impl_->memory_info,
            impl_->flat_output_buffer.data(),
            impl_->flat_output_buffer.size(),
            output_shape.data(),
            output_shape.size()
        );

        const char* input_names[] = { impl_->input_name.c_str() };
        const char* output_names[] = { impl_->output_name.c_str() };

        impl_->session->Run(
            impl_->run_options,
            input_names,
            &input_tensor,
            1,
            output_names,
            &output_tensor,
            1
        );

        const auto t_end = std::chrono::high_resolution_clock::now();
        const float total_us = static_cast<float>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()
        ) / 1000.0f;
        const float per_item_us = total_us / static_cast<float>(batch_size);

        decisions.reserve(batch_size);
        const float* out_ptr = impl_->flat_output_buffer.data();
        for (size_t i = 0; i < batch_size; ++i) {
            const float q0 = out_ptr[i * ACTION_DIM + 0];
            const float q1 = out_ptr[i * ACTION_DIM + 1];
            const int best_action = (q1 > q0) ? 1 : 0;
            decisions.push_back(RLActionDecision{ best_action, q0, q1, per_item_us });
        }

        last_latency_us_ = per_item_us;
        total_latency_us_ += total_us;
        inference_count_ += batch_size;
        last_error_.clear();
        return decisions;
    } catch (const Ort::Exception&) {
        // Fallback for static-batch ONNX models: evaluate sequentially
        decisions.clear();
        decisions.reserve(batch_size);
        std::vector<float> single_obs(OBSERVATION_DIM);
        for (size_t i = 0; i < batch_size; ++i) {
            std::copy_n(flat_observations + i * OBSERVATION_DIM, OBSERVATION_DIM, single_obs.data());
            decisions.push_back(predict(single_obs));
        }
        return decisions;
    }
}

std::vector<RLActionDecision> RLPolicyInference::predictBatch(
    const std::vector<std::vector<float>>& batch_observations
) {
    if (batch_observations.empty()) return {};
    if (batch_observations.size() == 1) {
        return { predict(batch_observations[0]) };
    }

    if (!isLoaded()) {
        last_error_ = "predictBatch() called on uninitialized or unloaded model.";
        return std::vector<RLActionDecision>(batch_observations.size(), RLActionDecision{ 0, 0.0f, 0.0f, 0.0f });
    }

    const size_t batch_size = batch_observations.size();
    impl_->flat_input_buffer.resize(batch_size * OBSERVATION_DIM);

    for (size_t i = 0; i < batch_size; ++i) {
        if (batch_observations[i].size() != OBSERVATION_DIM) {
            // Fall back to element-wise validation for mixed/invalid dimensions
            std::vector<RLActionDecision> decisions;
            decisions.reserve(batch_size);
            for (const auto& obs : batch_observations) {
                decisions.push_back(predict(obs));
            }
            return decisions;
        }
        std::copy_n(batch_observations[i].data(), OBSERVATION_DIM,
                    impl_->flat_input_buffer.data() + i * OBSERVATION_DIM);
    }

    return predictBatchFlat(impl_->flat_input_buffer.data(), batch_size);
}

std::string RLPolicyInference::getLastError() const {
    return last_error_;
}

float RLPolicyInference::getLastLatencyUs() const noexcept {
    return last_latency_us_;
}

float RLPolicyInference::getAverageLatencyUs() const noexcept {
    if (inference_count_ == 0) return 0.0f;
    return static_cast<float>(total_latency_us_ / static_cast<double>(inference_count_));
}

uint64_t RLPolicyInference::getInferenceCount() const noexcept {
    return inference_count_;
}

void RLPolicyInference::resetMetrics() noexcept {
    last_latency_us_  = 0.0f;
    total_latency_us_ = 0.0;
    inference_count_  = 0;
}

} // namespace syntraq
