//
// src/metrics/traffic_metrics.cpp
//
// Implementation of traffic metrics collection, aggregation, and CSV export.
//

#include "syntraq/metrics/traffic_metrics.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

namespace syntraq {

// ── RunResult Helpers ─────────────────────────────────────────────────────────

std::string RunResult::csv_header() {
    return "scenario,seed,duration_s,total_ticks,total_spawned,total_completed,"
           "active_remaining,avg_speed_mps,avg_speed_kmh,avg_waiting_time_s,"
           "avg_travel_time_s,max_queue_length,avg_queue_length,throughput_vps,"
           "throughput_vph,congestion_ratio";
}

std::string RunResult::to_csv_row() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << scenario_name << ","
       << seed << ","
       << duration_s << ","
       << total_ticks << ","
       << total_spawned << ","
       << total_completed << ","
       << active_remaining << ","
       << avg_speed_mps << ","
       << avg_speed_kmh << ","
       << avg_waiting_time_s << ","
       << avg_travel_time_s << ","
       << max_queue_length << ","
       << avg_queue_length << ","
       << throughput_vps << ","
       << throughput_vph << ","
       << congestion_ratio;
    return ss.str();
}

bool RunResult::export_csv(const std::string& filepath, bool append) const {
    std::ifstream check_file(filepath);
    const bool exists = check_file.good();
    check_file.close();

    std::ofstream out;
    if (append && exists) {
        out.open(filepath, std::ios::app);
    } else {
        out.open(filepath, std::ios::out | std::ios::trunc);
    }

    if (!out.is_open()) {
        return false;
    }

    if (!append || !exists) {
        out << csv_header() << "\n";
    }
    out << to_csv_row() << "\n";
    return true;
}

// ── TrafficMetricsCollector ───────────────────────────────────────────────────

TrafficMetricsCollector::TrafficMetricsCollector() {
    reset();
}

void TrafficMetricsCollector::set_scenario(TrafficScenario scenario, std::string scenario_name) {
    scenario_      = scenario;
    scenario_name_ = std::move(scenario_name);
}

void TrafficMetricsCollector::set_seed(uint32_t seed) {
    seed_ = seed;
}

void TrafficMetricsCollector::reset() {
    elapsed_s_       = 0.0f;
    total_ticks_     = 0;
    total_spawned_   = 0;
    total_completed_ = 0;

    completed_travel_time_s_ = 0.0;
    completed_wait_time_s_   = 0.0;
    completed_distance_m_    = 0.0;

    sum_speed_samples_    = 0.0;
    speed_sample_count_   = 0;
    sum_queued_vehicles_  = 0.0;
    sum_congestion_ratio_ = 0.0;
    peak_max_queue_       = 0;

    current_snapshot_ = MetricSnapshot{};
    timeseries_.clear();
}

void TrafficMetricsCollector::record_arrival(const Vehicle& arrived_vehicle) {
    total_completed_++;
    completed_travel_time_s_ += static_cast<double>(arrived_vehicle.travel_time_s);
    completed_wait_time_s_   += static_cast<double>(arrived_vehicle.wait_time_s);
    completed_distance_m_    += static_cast<double>(arrived_vehicle.distance_m);
}

void TrafficMetricsCollector::record_tick(float                       dt,
                                          const std::vector<Vehicle>& active_vehicles,
                                          uint32_t                    total_spawned_cumulative,
                                          uint64_t                    tick_number) {
    elapsed_s_     += dt;
    total_ticks_    = tick_number;
    total_spawned_  = total_spawned_cumulative;

    const uint32_t active_count = static_cast<uint32_t>(active_vehicles.size());
    uint32_t queued_count       = 0;
    double   sum_speeds         = 0.0;
    double   sum_waits          = 0.0;
    double   sum_travels        = 0.0;

    // Track lane queue depths for max queue detection
    std::map<std::pair<RoadId, uint32_t>, uint32_t> lane_queues;

    for (const auto& v : active_vehicles) {
        sum_speeds  += static_cast<double>(v.speed_mps);
        sum_waits   += static_cast<double>(v.wait_time_s);
        sum_travels += static_cast<double>(v.travel_time_s);

        const bool is_queued = (v.state == VehicleState::Stopped || v.speed_mps < 0.5f);
        if (is_queued) {
            queued_count++;
            lane_queues[{v.current_road, v.lane_index}]++;
        }
    }

    uint32_t current_max_lane_queue = 0;
    for (const auto& [lane, count] : lane_queues) {
        if (count > current_max_lane_queue) {
            current_max_lane_queue = count;
        }
    }

    if (current_max_lane_queue > peak_max_queue_) {
        peak_max_queue_ = current_max_lane_queue;
    }

    const float congestion_ratio = (active_count > 0)
        ? std::clamp(static_cast<float>(queued_count) / static_cast<float>(active_count), 0.0f, 1.0f)
        : 0.0f;

    // Accumulate time-averaged fleet metrics
    sum_speed_samples_    += sum_speeds;
    speed_sample_count_   += active_count;
    sum_queued_vehicles_  += static_cast<double>(queued_count);
    sum_congestion_ratio_ += static_cast<double>(congestion_ratio);

    // Build current snapshot
    current_snapshot_.tick             = total_ticks_;
    current_snapshot_.timestamp_s      = elapsed_s_;
    current_snapshot_.active_vehicles  = active_count;
    current_snapshot_.queued_vehicles  = queued_count;
    current_snapshot_.total_spawned    = total_spawned_;
    current_snapshot_.total_completed  = total_completed_;
    current_snapshot_.avg_speed_mps    = (active_count > 0)
        ? static_cast<float>(sum_speeds / active_count)
        : 0.0f;
    current_snapshot_.avg_waiting_time_s = (active_count > 0)
        ? static_cast<float>(sum_waits / active_count)
        : (total_completed_ > 0 ? static_cast<float>(completed_wait_time_s_ / total_completed_) : 0.0f);
    current_snapshot_.avg_travel_time_s = (active_count > 0)
        ? static_cast<float>(sum_travels / active_count)
        : (total_completed_ > 0 ? static_cast<float>(completed_travel_time_s_ / total_completed_) : 0.0f);
    current_snapshot_.max_queue_length = current_max_lane_queue;
    current_snapshot_.throughput_vph   = (elapsed_s_ > 0.0f)
        ? (static_cast<float>(total_completed_) / elapsed_s_) * 3600.0f
        : 0.0f;
    current_snapshot_.congestion_ratio = congestion_ratio;

    timeseries_.push_back(current_snapshot_);
}

RunResult TrafficMetricsCollector::get_run_result() const {
    RunResult result;
    result.scenario         = scenario_;
    result.scenario_name    = scenario_name_;
    result.seed             = seed_;
    result.duration_s       = elapsed_s_;
    result.total_ticks      = total_ticks_;
    result.total_spawned    = total_spawned_;
    result.total_completed  = total_completed_;
    result.active_remaining = current_snapshot_.active_vehicles;

    // Average travel time
    if (total_completed_ > 0) {
        result.avg_travel_time_s = static_cast<float>(completed_travel_time_s_ / total_completed_);
    } else {
        result.avg_travel_time_s = current_snapshot_.avg_travel_time_s;
    }

    // Average waiting time
    if (total_completed_ > 0) {
        result.avg_waiting_time_s = static_cast<float>(completed_wait_time_s_ / total_completed_);
    } else {
        result.avg_waiting_time_s = current_snapshot_.avg_waiting_time_s;
    }

    // Average speed
    if (speed_sample_count_ > 0) {
        result.avg_speed_mps = static_cast<float>(sum_speed_samples_ / speed_sample_count_);
    } else if (total_completed_ > 0 && completed_travel_time_s_ > 0.0) {
        result.avg_speed_mps = static_cast<float>(completed_distance_m_ / completed_travel_time_s_);
    } else {
        result.avg_speed_mps = 0.0f;
    }
    result.avg_speed_kmh = result.avg_speed_mps * 3.6f;

    // Queue metrics
    result.max_queue_length = peak_max_queue_;
    result.avg_queue_length = (total_ticks_ > 0)
        ? static_cast<float>(sum_queued_vehicles_ / static_cast<double>(total_ticks_))
        : 0.0f;

    // Throughput
    result.throughput_vps = (elapsed_s_ > 0.0f)
        ? (static_cast<float>(total_completed_) / elapsed_s_)
        : 0.0f;
    result.throughput_vph = result.throughput_vps * 3600.0f;

    // Congestion ratio
    result.congestion_ratio = (total_ticks_ > 0)
        ? static_cast<float>(sum_congestion_ratio_ / static_cast<double>(total_ticks_))
        : 0.0f;

    return result;
}

bool TrafficMetricsCollector::export_summary_csv(const std::string& filepath, bool append) const {
    return get_run_result().export_csv(filepath, append);
}

bool TrafficMetricsCollector::export_timeseries_csv(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }

    out << "tick,timestamp_s,active_vehicles,queued_vehicles,total_spawned,total_completed,"
        << "avg_speed_mps,avg_waiting_time_s,avg_travel_time_s,max_queue_length,"
        << "throughput_vph,congestion_ratio\n";

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    for (const auto& s : timeseries_) {
        ss.str("");
        ss.clear();
        ss << s.tick << ","
           << s.timestamp_s << ","
           << s.active_vehicles << ","
           << s.queued_vehicles << ","
           << s.total_spawned << ","
           << s.total_completed << ","
           << s.avg_speed_mps << ","
           << s.avg_waiting_time_s << ","
           << s.avg_travel_time_s << ","
           << s.max_queue_length << ","
           << s.throughput_vph << ","
           << s.congestion_ratio << "\n";
        out << ss.str();
    }
    return true;
}

} // namespace syntraq
