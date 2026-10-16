//
// src/metrics/traffic_data_collector.cpp
//

#include "syntraq/metrics/traffic_data_collector.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>

namespace syntraq {

// ── TrafficDataSample CSV Serialization ───────────────────────────────────────

std::string TrafficDataSample::csv_header() {
    return "run_id,scenario_id,scenario_name,seed,timestamp_s,sample_idx,"
           "vehicle_count,total_spawned,total_arrived,spawn_rate_vpm,"
           "queue_vehicle_count,max_queue_length,avg_queue_per_lane,queue_ratio,"
           "avg_speed_mps,avg_speed_kmh,speed_ratio,avg_wait_time_s,avg_travel_time_s,"
           "road_density_vpm,road_occupancy_pct,critical_occupancy_pct,"
           "signal_phase_index,time_in_phase_s,phase_duration_s,phase_elapsed_ratio,is_all_red,green_approaches_count,"
           "recent_throughput_vph,recent_queue_delta,recent_speed_delta,"
           "has_target_15s,target_queue_15s,target_max_queue_15s,target_congestion_15s,"
           "has_target_30s,target_queue_30s,target_max_queue_30s,target_congestion_30s";
}

std::string TrafficDataSample::to_csv_row() const {
    std::ostringstream ss;
    ss << std::fixed;

    // Metadata
    ss << run_id << ","
       << scenario_id << ","
       << scenario_name << ","
       << seed << ","
       << std::setprecision(2) << timestamp_s << ","
       << sample_idx << ",";

    // Counts
    ss << vehicle_count << ","
       << total_spawned << ","
       << total_arrived << ","
       << std::setprecision(2) << spawn_rate_vpm << ",";

    // Queues
    ss << queue_vehicle_count << ","
       << max_queue_length << ","
       << std::setprecision(3) << avg_queue_per_lane << ","
       << std::setprecision(4) << queue_ratio << ",";

    // Speeds & Travel times
    ss << std::setprecision(2) << avg_speed_mps << ","
       << std::setprecision(2) << avg_speed_kmh << ","
       << std::setprecision(4) << speed_ratio << ","
       << std::setprecision(2) << avg_wait_time_s << ","
       << std::setprecision(2) << avg_travel_time_s << ",";

    // Occupancy
    ss << std::setprecision(5) << road_density_vpm << ","
       << std::setprecision(2) << road_occupancy_pct << ","
       << std::setprecision(2) << critical_occupancy_pct << ",";

    // Signals
    ss << signal_phase_index << ","
       << std::setprecision(2) << time_in_phase_s << ","
       << std::setprecision(2) << phase_duration_s << ","
       << std::setprecision(4) << phase_elapsed_ratio << ","
       << is_all_red << ","
       << green_approaches_count << ",";

    // Recent momentum
    ss << std::setprecision(1) << recent_throughput_vph << ","
       << std::setprecision(3) << recent_queue_delta << ","
       << std::setprecision(3) << recent_speed_delta << ",";

    // Targets
    ss << (has_target_15s ? 1 : 0) << ","
       << target_queue_15s << ","
       << target_max_queue_15s << ","
       << std::setprecision(4) << target_congestion_15s << ","
       << (has_target_30s ? 1 : 0) << ","
       << target_queue_30s << ","
       << target_max_queue_30s << ","
       << std::setprecision(4) << target_congestion_30s;

    return ss.str();
}

// ── TrafficDataCollector Implementation ───────────────────────────────────────

TrafficDataCollector::TrafficDataCollector(float sample_interval_s,
                                           float horizon_1_s,
                                           float horizon_2_s)
    : sample_interval_s_(sample_interval_s)
    , horizon_1_s_      (horizon_1_s)
    , horizon_2_s_      (horizon_2_s)
{
}

void TrafficDataCollector::reset() {
    last_sample_time_ = -999.0f;
    next_sample_idx_  = 0;
    rolling_history_.clear();
    samples_.clear();
    pending_h1_indices_.clear();
    pending_h2_indices_.clear();
}

void TrafficDataCollector::set_run_metadata(std::string     run_id,
                                            TrafficScenario scenario,
                                            std::string     scenario_name,
                                            uint32_t        seed) {
    run_id_        = std::move(run_id);
    scenario_      = scenario;
    scenario_name_ = std::move(scenario_name);
    seed_          = seed;
}

void TrafficDataCollector::update(float                                                           sim_time_s,
                                  const RoadNetwork&                                             network,
                                  const std::vector<Vehicle>&                                    vehicles,
                                  const std::unordered_map<IntersectionId, TrafficSignalController>& controllers,
                                  uint32_t                                                       total_spawned,
                                  uint32_t                                                       total_arrived,
                                  float                                                          spawn_interval_s) {
    // ── 1. Calculate Instantaneous Metrics at sim_time_s ─────────────────────
    uint32_t active_count  = 0;
    uint32_t queued_count  = 0;
    float    sum_speed_mps = 0.0f;
    float    sum_wait_s    = 0.0f;
    float    sum_travel_s  = 0.0f;

    std::map<std::pair<RoadId, uint32_t>, uint32_t> lane_queues;
    std::map<RoadId, float> road_vehicle_lengths;

    for (const auto& v : vehicles) {
        if (v.state == VehicleState::Arrived) continue;

        ++active_count;
        sum_speed_mps += v.speed_mps;
        sum_wait_s    += v.wait_time_s;
        sum_travel_s  += v.travel_time_s;

        road_vehicle_lengths[v.current_road] += v.length_m;

        if (v.state == VehicleState::Stopped || v.speed_mps < 0.5f) {
            ++queued_count;
            lane_queues[{ v.current_road, v.lane_index }]++;
        }
    }

    uint32_t max_queue = 0;
    for (const auto& [lane_key, count] : lane_queues) {
        (void)lane_key;
        max_queue = std::max(max_queue, count);
    }

    const float cur_avg_speed = active_count > 0 ? (sum_speed_mps / static_cast<float>(active_count)) : 0.0f;
    const float cur_congestion = active_count > 0 ? (static_cast<float>(queued_count) / static_cast<float>(active_count)) : 0.0f;

    // ── 2. Resolve Ground Truth Targets for Prior Samples ────────────────────
    while (!pending_h1_indices_.empty()) {
        const size_t idx = pending_h1_indices_.front();
        if (samples_[idx].timestamp_s + horizon_1_s_ <= sim_time_s + 1e-4f) {
            samples_[idx].has_target_15s       = true;
            samples_[idx].target_queue_15s     = queued_count;
            samples_[idx].target_max_queue_15s = max_queue;
            samples_[idx].target_congestion_15s = cur_congestion;
            pending_h1_indices_.pop_front();
        } else {
            break;
        }
    }

    while (!pending_h2_indices_.empty()) {
        const size_t idx = pending_h2_indices_.front();
        if (samples_[idx].timestamp_s + horizon_2_s_ <= sim_time_s + 1e-4f) {
            samples_[idx].has_target_30s       = true;
            samples_[idx].target_queue_30s     = queued_count;
            samples_[idx].target_max_queue_30s = max_queue;
            samples_[idx].target_congestion_30s = cur_congestion;
            pending_h2_indices_.pop_front();
        } else {
            break;
        }
    }

    // ── 3. Check Sampling Interval ───────────────────────────────────────────
    if (sim_time_s - last_sample_time_ < sample_interval_s_ - 1e-4f) {
        return;
    }
    last_sample_time_ = sim_time_s;

    // ── 4. Calculate Network Topology Occupancies ─────────────────────────────
    float total_road_length_m = 0.0f;
    float total_lane_length_m = 0.0f;
    uint32_t total_lanes      = 0;
    float max_road_occ_pct    = 0.0f;

    for (const auto& [road_id, road] : network.roads()) {
        const float r_len = road.length_m;
        const uint32_t r_lanes = std::max(1u, road.lane_count());
        total_road_length_m += r_len;
        total_lane_length_m += r_len * static_cast<float>(r_lanes);
        total_lanes += r_lanes;

        auto it = road_vehicle_lengths.find(road_id);
        if (it != road_vehicle_lengths.end() && r_len > 0.0f) {
            const float r_occ = (it->second / (r_len * static_cast<float>(r_lanes))) * 100.0f;
            max_road_occ_pct = std::max(max_road_occ_pct, std::min(100.0f, r_occ));
        }
    }

    float total_vehicle_length_m = 0.0f;
    for (const auto& [road_id, len] : road_vehicle_lengths) {
        (void)road_id;
        total_vehicle_length_m += len;
    }

    const float net_density_vpm = total_road_length_m > 0.0f
        ? (static_cast<float>(active_count) / total_road_length_m)
        : 0.0f;

    const float net_occupancy_pct = total_lane_length_m > 0.0f
        ? std::min(100.0f, (total_vehicle_length_m / total_lane_length_m) * 100.0f)
        : 0.0f;

    const float avg_queue_lane = total_lanes > 0
        ? (static_cast<float>(queued_count) / static_cast<float>(total_lanes))
        : 0.0f;

    const float speed_ratio = cur_avg_speed > 0.0f ? std::min(1.0f, cur_avg_speed / 13.89f) : 0.0f;

    // ── 5. Signal State Aggregation ──────────────────────────────────────────
    int32_t phase_idx      = -1;
    float   phase_elapsed  = 0.0f;
    float   phase_duration = 0.0f;
    float   phase_ratio    = 0.0f;
    int32_t is_all_red     = 0;
    int32_t green_lanes    = 0;

    if (!controllers.empty()) {
        // Sample primary signal controller
        const auto& ctrl = controllers.begin()->second;
        phase_idx        = static_cast<int32_t>(ctrl.current_phase_index());
        phase_elapsed    = ctrl.stage_elapsed_s();
        phase_duration   = ctrl.current_stage_duration();
        phase_ratio      = phase_duration > 0.0f ? std::min(1.0f, phase_elapsed / phase_duration) : 0.0f;
        is_all_red       = (ctrl.current_stage() == PhaseStage::AllRed) ? 1 : 0;

        const auto* phase_ptr = ctrl.current_phase();
        if (phase_ptr && ctrl.current_stage() == PhaseStage::Green) {
            green_lanes = static_cast<int32_t>(phase_ptr->green_roads.size());
        } else {
            green_lanes = 0;
        }
    }

    // ── 6. Rolling Window Temporal Momentum ──────────────────────────────────
    rolling_history_.push_back({ sim_time_s, queued_count, cur_avg_speed, total_arrived });

    constexpr float kWindowS = 10.0f;
    while (rolling_history_.size() > 1 &&
           (sim_time_s - rolling_history_.front().timestamp_s > kWindowS)) {
        rolling_history_.pop_front();
    }

    float recent_throughput = 0.0f;
    float recent_q_delta    = 0.0f;
    float recent_v_delta    = 0.0f;

    if (rolling_history_.size() >= 2) {
        const auto& oldest = rolling_history_.front();
        const float dt = sim_time_s - oldest.timestamp_s;
        if (dt > 0.1f) {
            const float d_arrived = static_cast<float>(total_arrived - oldest.total_arrived);
            recent_throughput = (d_arrived / dt) * 3600.0f;
            recent_q_delta    = static_cast<float>(static_cast<int32_t>(queued_count) - static_cast<int32_t>(oldest.queue_count)) / dt;
            recent_v_delta    = (cur_avg_speed - oldest.avg_speed_mps) / dt;
        }
    }

    const float spawn_rate_vpm = spawn_interval_s > 0.0f ? (60.0f / spawn_interval_s) : 0.0f;

    // ── 7. Construct & Store Sample ──────────────────────────────────────────
    TrafficDataSample sample;
    sample.run_id                 = run_id_;
    sample.scenario_id            = static_cast<uint32_t>(scenario_);
    sample.scenario_name          = scenario_name_;
    sample.seed                   = seed_;
    sample.timestamp_s            = sim_time_s;
    sample.sample_idx             = next_sample_idx_++;

    sample.vehicle_count          = active_count;
    sample.total_spawned          = total_spawned;
    sample.total_arrived          = total_arrived;
    sample.spawn_rate_vpm         = spawn_rate_vpm;

    sample.queue_vehicle_count    = queued_count;
    sample.max_queue_length       = max_queue;
    sample.avg_queue_per_lane     = avg_queue_lane;
    sample.queue_ratio            = cur_congestion;

    sample.avg_speed_mps          = cur_avg_speed;
    sample.avg_speed_kmh          = cur_avg_speed * 3.6f;
    sample.speed_ratio            = speed_ratio;
    sample.avg_wait_time_s        = active_count > 0 ? (sum_wait_s / static_cast<float>(active_count)) : 0.0f;
    sample.avg_travel_time_s      = active_count > 0 ? (sum_travel_s / static_cast<float>(active_count)) : 0.0f;

    sample.road_density_vpm       = net_density_vpm;
    sample.road_occupancy_pct     = net_occupancy_pct;
    sample.critical_occupancy_pct = max_road_occ_pct;

    sample.signal_phase_index     = phase_idx;
    sample.time_in_phase_s        = phase_elapsed;
    sample.phase_duration_s       = phase_duration;
    sample.phase_elapsed_ratio    = phase_ratio;
    sample.is_all_red             = is_all_red;
    sample.green_approaches_count = green_lanes;

    sample.recent_throughput_vph  = recent_throughput;
    sample.recent_queue_delta     = recent_q_delta;
    sample.recent_speed_delta     = recent_v_delta;

    const size_t sample_pos = samples_.size();
    samples_.push_back(std::move(sample));

    pending_h1_indices_.push_back(sample_pos);
    pending_h2_indices_.push_back(sample_pos);
}

void TrafficDataCollector::finalize() {
    // If simulation ended, any sample whose target horizon falls past the run duration
    // retains has_target_* = false. This guarantees no false or fabricated targets.
    pending_h1_indices_.clear();
    pending_h2_indices_.clear();
}

std::vector<TrafficDataSample> TrafficDataCollector::valid_samples(bool require_30s) const {
    std::vector<TrafficDataSample> result;
    result.reserve(samples_.size());

    for (const auto& s : samples_) {
        if (require_30s) {
            if (s.has_target_30s) result.push_back(s);
        } else {
            if (s.has_target_15s) result.push_back(s);
        }
    }
    return result;
}

bool TrafficDataCollector::export_csv(const std::string& filepath,
                                      bool               append,
                                      bool               require_30s) const {
    const bool exists = std::ifstream(filepath).good();

    std::ios_base::openmode mode = std::ios::out;
    if (append && exists) {
        mode |= std::ios::app;
    } else {
        mode |= std::ios::trunc;
    }

    std::ofstream file(filepath, mode);
    if (!file.is_open()) return false;

    if (!append || !exists) {
        file << TrafficDataSample::csv_header() << "\n";
    }

    for (const auto& s : samples_) {
        if (require_30s && !s.has_target_30s) continue;
        if (!require_30s && !s.has_target_15s) continue;

        file << s.to_csv_row() << "\n";
    }

    return file.good();
}

} // namespace syntraq
