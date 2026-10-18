#pragma once

//
// syntraq/ai/traffic_feature_scaler.hpp
//
// Preprocessing engine for native traffic prediction inference.
// Standardizes raw continuous simulation telemetry and one-hot encodes
// categorical scenario demand regimes.
// Matches Python scikit-learn / pandas training preprocessing exactly.
//

#include "syntraq/metrics/traffic_data_collector.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace syntraq {

struct FeatureScaler {
    float mean { 0.0f };
    float std  { 1.0f };

    [[nodiscard]] constexpr float scale(float val) const noexcept {
        return (val - mean) / (std > 1e-6f ? std : 1.0f);
    }
};

/// Preprocessing pipeline ensuring C++ inference inputs mirror Python training data.
class TrafficFeatureScaler {
public:
    static constexpr std::size_t NUM_CONTINUOUS_FEATURES = 22;
    static constexpr std::size_t NUM_SCENARIOS           = 4;
    static constexpr std::size_t TOTAL_FEATURES          = 26;

    TrafficFeatureScaler();

    /// Transform a structured simulation observation into a 26-dimensional model input vector.
    [[nodiscard]] std::vector<float> transform(
        const TrafficDataSample& sample,
        const std::string& scenario_name = ""
    ) const;

    /// Transform raw continuous features vector (22 dims) plus scenario name into 26 dims.
    [[nodiscard]] std::vector<float> transformRaw(
        const std::vector<float>& raw_22,
        const std::string& scenario_name = ""
    ) const;

    /// Override scalers from a JSON string (e.g. from metadata.json).
    bool loadFromJsonString(const std::string& json_str);

    /// Override scalers from a JSON file path.
    bool loadFromJsonFile(const std::string& file_path);

    [[nodiscard]] const FeatureScaler& getScaler(const std::string& feature_name) const;

    [[nodiscard]] static const std::vector<std::string>& getFeatureNames() noexcept;

private:
    void initDefaultScalers();

    std::unordered_map<std::string, FeatureScaler> scalers_;
    static const FeatureScaler kDefaultFallback;
};

} // namespace syntraq
