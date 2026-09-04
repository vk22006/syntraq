//
// src/core/config.cpp
//
// Config implementation.
// Uses only the C++ standard library — no external JSON dependency yet.
// The from_file() stub prints a warning and returns defaults; a real parser
// will be wired in a future milestone when it's actually needed.
//

#include "syntraq/core/config.h"

#include <fstream>
#include <iostream>

namespace syntraq {

Config Config::from_file(const std::string& path) {
    // Milestone 1: JSON parsing is not yet implemented.
    // Return defaults and emit a diagnostic so the caller knows.
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[Config] Could not open '" << path
                  << "' — using defaults.\n";
    } else {
        std::cerr << "[Config] JSON parsing not yet implemented; "
                     "using defaults.\n";
    }
    return Config{};
}

} // namespace syntraq
