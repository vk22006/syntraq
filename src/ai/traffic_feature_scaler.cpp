//
// syntraq/ai/traffic_feature_scaler.cpp
//
// Implementation of preprocessing transformation pipeline for native AI inference.
//

#include "syntraq/ai/traffic_feature_scaler.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace syntraq {

const FeatureScaler TrafficFeatureScaler::kDefaultFallback{ 0.0f, 1.0f };

const std::vector<std::string>& TrafficFeatureScaler::getFeatureNames() noexcept {
    static const std::vector<std::string> kNames = {
        "vehicle_count",
        "spawn_rate_vpm",
        "queue_vehicle_count",
        "max_queue_length",
        "avg_queue_per_lane",
        "queue_ratio",
        "avg_speed_mps",
        "avg_speed_kmh",
        "speed_ratio",
        "avg_wait_time_s",
        "avg_travel_time_s",
        "road_density_vpm",
        "road_occupancy_pct",
        "critical_occupancy_pct",
        "time_in_phase_s",
        "phase_duration_s",
        "phase_elapsed_ratio",
        "is_all_red",
        "green_approaches_count",
        "recent_throughput_vph",
        "recent_queue_delta",
        "recent_speed_delta",
        "scenario_low_traffic",
        "scenario_medium_traffic",
        "scenario_high_traffic",
        "scenario_rush_hour"
    };
    return kNames;
}

TrafficFeatureScaler::TrafficFeatureScaler() {
    initDefaultScalers();
}

void TrafficFeatureScaler::initDefaultScalers() {
    // Exact scaler statistics fitted strictly on training partition (from metadata.json)
    scalers_["vehicle_count"]          = { 30.4324074f, 26.8689723f };
    scalers_["spawn_rate_vpm"]         = { 56.2500000f, 40.2329005f };
    scalers_["queue_vehicle_count"]    = {  5.2287037f,  6.9260967f };
    scalers_["max_queue_length"]       = {  1.1120370f,  0.9180728f };
    scalers_["avg_queue_per_lane"]     = {  0.0769269f,  0.1018368f };
    scalers_["queue_ratio"]            = {  0.1467330f,  0.1530953f };
    scalers_["avg_speed_mps"]          = {  8.7768426f,  2.0884901f };
    scalers_["avg_speed_kmh"]          = { 31.5967778f,  7.5184002f };
    scalers_["speed_ratio"]            = {  0.6318881f,  0.1503548f };
    scalers_["avg_wait_time_s"]        = {  2.4560926f,  2.1343841f };
    scalers_["avg_travel_time_s"]      = { 19.4412315f,  9.9622268f };
    scalers_["road_density_vpm"]       = {  0.0049723f,  0.0043907f };
    scalers_["road_occupancy_pct"]     = {  1.1186481f,  0.9881750f };
    scalers_["critical_occupancy_pct"] = {  4.4155093f,  2.7716379f };
    scalers_["time_in_phase_s"]        = {  3.3666667f,  2.9558381f };
    scalers_["phase_duration_s"]       = {  7.5333333f,  3.5014627f };
    scalers_["phase_elapsed_ratio"]    = {  0.4200000f,  0.2859736f };
    scalers_["is_all_red"]             = {  0.1333333f,  0.3400921f };
    scalers_["green_approaches_count"] = {  0.6666667f,  0.4716229f };
    scalers_["recent_throughput_vph"]  = { 1017.00843f, 1388.76008f };
    scalers_["recent_queue_delta"]     = {  0.0805056f,  0.6744245f };
    scalers_["recent_speed_delta"]     = {  0.1890065f,  0.6038578f };
}

const FeatureScaler& TrafficFeatureScaler::getScaler(const std::string& feature_name) const {
    auto it = scalers_.find(feature_name);
    if (it != scalers_.end()) {
        return it->second;
    }
    return kDefaultFallback;
}

std::vector<float> TrafficFeatureScaler::transform(
    const TrafficDataSample& sample,
    const std::string& scenario_name
) const {
    std::vector<float> out;
    out.reserve(TOTAL_FEATURES);

    // 1. Continuous features (scaled using fitted mean and std)
    out.push_back(getScaler("vehicle_count").scale(static_cast<float>(sample.vehicle_count)));
    out.push_back(getScaler("spawn_rate_vpm").scale(sample.spawn_rate_vpm));
    out.push_back(getScaler("queue_vehicle_count").scale(static_cast<float>(sample.queue_vehicle_count)));
    out.push_back(getScaler("max_queue_length").scale(static_cast<float>(sample.max_queue_length)));
    out.push_back(getScaler("avg_queue_per_lane").scale(sample.avg_queue_per_lane));
    out.push_back(getScaler("queue_ratio").scale(sample.queue_ratio));
    out.push_back(getScaler("avg_speed_mps").scale(sample.avg_speed_mps));
    out.push_back(getScaler("avg_speed_kmh").scale(sample.avg_speed_kmh));
    out.push_back(getScaler("speed_ratio").scale(sample.speed_ratio));
    out.push_back(getScaler("avg_wait_time_s").scale(sample.avg_wait_time_s));
    out.push_back(getScaler("avg_travel_time_s").scale(sample.avg_travel_time_s));
    out.push_back(getScaler("road_density_vpm").scale(sample.road_density_vpm));
    out.push_back(getScaler("road_occupancy_pct").scale(sample.road_occupancy_pct));
    out.push_back(getScaler("critical_occupancy_pct").scale(sample.critical_occupancy_pct));
    out.push_back(getScaler("time_in_phase_s").scale(sample.time_in_phase_s));
    out.push_back(getScaler("phase_duration_s").scale(sample.phase_duration_s));
    out.push_back(getScaler("phase_elapsed_ratio").scale(sample.phase_elapsed_ratio));
    out.push_back(getScaler("is_all_red").scale(static_cast<float>(sample.is_all_red)));
    out.push_back(getScaler("green_approaches_count").scale(static_cast<float>(sample.green_approaches_count)));
    out.push_back(getScaler("recent_throughput_vph").scale(sample.recent_throughput_vph));
    out.push_back(getScaler("recent_queue_delta").scale(sample.recent_queue_delta));
    out.push_back(getScaler("recent_speed_delta").scale(sample.recent_speed_delta));

    // 2. Categorical scenario one-hot features (Low, Med, High, Rush)
    std::string sc = scenario_name.empty() ? sample.scenario_name : scenario_name;
    // Lowercase conversion for case-insensitive matching
    std::string lower_sc;
    lower_sc.reserve(sc.size());
    for (char c : sc) {
        lower_sc.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    float is_low  = 0.0f;
    float is_med  = 0.0f;
    float is_high = 0.0f;
    float is_rush = 0.0f;

    if (!lower_sc.empty()) {
        if (lower_sc.find("low") != std::string::npos) {
            is_low = 1.0f;
        } else if (lower_sc.find("med") != std::string::npos) {
            is_med = 1.0f;
        } else if (lower_sc.find("high") != std::string::npos) {
            is_high = 1.0f;
        } else if (lower_sc.find("rush") != std::string::npos) {
            is_rush = 1.0f;
        }
    } else {
        if (sample.scenario_id == 0) is_low = 1.0f;
        else if (sample.scenario_id == 1) is_med = 1.0f;
        else if (sample.scenario_id == 2) is_high = 1.0f;
        else if (sample.scenario_id == 3) is_rush = 1.0f;
    }

    // Default to medium if ambiguous
    if (is_low == 0.0f && is_med == 0.0f && is_high == 0.0f && is_rush == 0.0f) {
        is_med = 1.0f;
    }

    out.push_back(is_low);
    out.push_back(is_med);
    out.push_back(is_high);
    out.push_back(is_rush);

    return out;
}

std::vector<float> TrafficFeatureScaler::transformRaw(
    const std::vector<float>& raw_22,
    const std::string& scenario_name
) const {
    std::vector<float> out;
    out.reserve(TOTAL_FEATURES);

    const auto& names = getFeatureNames();
    for (std::size_t i = 0; i < NUM_CONTINUOUS_FEATURES && i < raw_22.size(); ++i) {
        out.push_back(getScaler(names[i]).scale(raw_22[i]));
    }

    std::string lower_sc;
    lower_sc.reserve(scenario_name.size());
    for (char c : scenario_name) {
        lower_sc.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    float is_low  = (lower_sc.find("low") != std::string::npos) ? 1.0f : 0.0f;
    float is_med  = (lower_sc.find("med") != std::string::npos) ? 1.0f : 0.0f;
    float is_high = (lower_sc.find("high") != std::string::npos) ? 1.0f : 0.0f;
    float is_rush = (lower_sc.find("rush") != std::string::npos) ? 1.0f : 0.0f;

    if (is_low == 0.0f && is_med == 0.0f && is_high == 0.0f && is_rush == 0.0f) {
        is_med = 1.0f;
    }

    out.push_back(is_low);
    out.push_back(is_med);
    out.push_back(is_high);
    out.push_back(is_rush);

    return out;
}

bool TrafficFeatureScaler::loadFromJsonString(const std::string& json_str) {
    // Lightweight parser extracting scalers object without external dependency
    try {
        const auto& names = getFeatureNames();
        for (std::size_t i = 0; i < NUM_CONTINUOUS_FEATURES; ++i) {
            const std::string& name = names[i];
            auto key_pos = json_str.find("\"" + name + "\"");
            if (key_pos == std::string::npos) continue;

            auto mean_pos = json_str.find("\"mean\":", key_pos);
            auto std_pos  = json_str.find("\"std\":", key_pos);
            if (mean_pos != std::string::npos && std_pos != std::string::npos) {
                float mean_val = std::stof(json_str.substr(mean_pos + 7));
                float std_val  = std::stof(json_str.substr(std_pos + 6));
                scalers_[name] = { mean_val, std_val };
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool TrafficFeatureScaler::loadFromJsonFile(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) return false;
    std::stringstream buffer;
    buffer << file.rdbuf();
    return loadFromJsonString(buffer.str());
}

} // namespace syntraq
