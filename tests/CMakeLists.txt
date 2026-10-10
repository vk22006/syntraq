cmake_minimum_required(VERSION 3.25)

# ---------------------------------------------------------------------------
# GoogleTest — fetched via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.16.0
    GIT_SHALLOW    TRUE
)
# Prevent GTest from overriding our compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# ---------------------------------------------------------------------------
# Helper macro to register a test executable with CTest
# ---------------------------------------------------------------------------
function(syntraq_add_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})
    target_link_libraries(${TEST_NAME}
        PRIVATE syntraq_core GTest::gtest_main
    )
    target_compile_options(${TEST_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /EHsc>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
    )
    gtest_discover_tests(${TEST_NAME}
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 30
    )
endfunction()

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------
syntraq_add_test(test_config       test_config.cpp)
syntraq_add_test(test_simulation   test_simulation.cpp)
syntraq_add_test(test_road_network test_road_network.cpp)
syntraq_add_test(test_vehicles     test_vehicles.cpp)
syntraq_add_test(test_signals          test_signals.cpp)
syntraq_add_test(test_routing          test_routing.cpp)
syntraq_add_test(test_traffic_behavior test_traffic_behavior.cpp)
