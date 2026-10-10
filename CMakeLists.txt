cmake_minimum_required(VERSION 3.25)
project(SyntraQ VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++20 globally
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build type default
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---------------------------------------------------------------------------
# Raylib — imported target (cross-platform)
# ---------------------------------------------------------------------------
set(RAYLIB_SDK_DIR "D:/Game development projects/raylib-6.0_winarm64_msvc16"
    CACHE PATH "Path to the pre-built Raylib SDK")

if(WIN32)
    add_library(raylib SHARED IMPORTED GLOBAL)
    set_target_properties(raylib PROPERTIES
        IMPORTED_IMPLIB   "${RAYLIB_SDK_DIR}/lib/raylibdll.lib"
        IMPORTED_LOCATION "${RAYLIB_SDK_DIR}/lib/raylib.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${RAYLIB_SDK_DIR}/include"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${RAYLIB_SDK_DIR}/include"
    )
else()
    if(EXISTS "${RAYLIB_SDK_DIR}/lib/libraylib.a")
        add_library(raylib STATIC IMPORTED GLOBAL)
        set_target_properties(raylib PROPERTIES
            IMPORTED_LOCATION "${RAYLIB_SDK_DIR}/lib/libraylib.a"
            INTERFACE_INCLUDE_DIRECTORIES "${RAYLIB_SDK_DIR}/include"
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${RAYLIB_SDK_DIR}/include"
        )
        target_link_libraries(raylib INTERFACE m pthread dl GL rt X11)
    elseif(EXISTS "${RAYLIB_SDK_DIR}/lib/libraylib.so")
        add_library(raylib SHARED IMPORTED GLOBAL)
        set_target_properties(raylib PROPERTIES
            IMPORTED_LOCATION "${RAYLIB_SDK_DIR}/lib/libraylib.so"
            INTERFACE_INCLUDE_DIRECTORIES "${RAYLIB_SDK_DIR}/include"
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${RAYLIB_SDK_DIR}/include"
        )
        target_link_libraries(raylib INTERFACE m pthread dl GL rt X11)
    else()
        message(FATAL_ERROR "Could not find libraylib.a or libraylib.so in ${RAYLIB_SDK_DIR}/lib")
    endif()
endif()

# ---------------------------------------------------------------------------
# Dear ImGui — fetched via FetchContent (header + source, no prebuilt)
# rlImGui requires imgui.h and the imgui backends; we compile imgui from source
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.92.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

# Build ImGui as a static library (core + OpenGL3 backend not needed —
# rlImGui supplies its own raylib backend)
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
)
target_include_directories(imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR})
# Suppress MSVC warnings in third-party imgui source
target_compile_options(imgui PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W0>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>
)

# ---------------------------------------------------------------------------
# rlImGui — compile from local SDK if present, otherwise fetch via FetchContent
# ---------------------------------------------------------------------------
if(EXISTS "${RAYLIB_SDK_DIR}/rlImGui-Raylib_6_0/rlImGui.cpp")
    set(RLIMGUI_DIR "${RAYLIB_SDK_DIR}/rlImGui-Raylib_6_0")
else()
    FetchContent_Declare(
        rlimgui_repo
        GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
        GIT_TAG        Raylib_6_0
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(rlimgui_repo)
    if(NOT rlimgui_repo_POPULATED)
        FetchContent_Populate(rlimgui_repo)
    endif()
    set(RLIMGUI_DIR "${rlimgui_repo_SOURCE_DIR}")
endif()

add_library(rlimgui STATIC
    "${RLIMGUI_DIR}/rlImGui.cpp"
)
target_include_directories(rlimgui SYSTEM PUBLIC
    "${RLIMGUI_DIR}"
    ${imgui_SOURCE_DIR}
)
target_link_libraries(rlimgui PUBLIC imgui raylib)
target_compile_options(rlimgui PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W0>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>
)

# ---------------------------------------------------------------------------
# syntraq_core — headless simulation library (no main, fully testable)
# ---------------------------------------------------------------------------
add_library(syntraq_core STATIC
    src/core/config.cpp
    src/simulation/simulation.cpp
    src/world/road_network.cpp
    src/vehicles/vehicle_movement_system.cpp
    src/vehicles/vehicle_spawner.cpp
    src/vehicles/route_provider.cpp
    src/signals/traffic_light.cpp
    src/signals/traffic_signal_controller.cpp
    src/routing/dijkstra_router.cpp
    src/routing/astar_router.cpp
    src/routing/graph_generator.cpp
    src/metrics/traffic_metrics.cpp
)
target_include_directories(syntraq_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
# Warning flags for our own code
target_compile_options(syntraq_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /EHsc>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

# ---------------------------------------------------------------------------
# syntraq_renderer — rendering layer (depends on raylib + imgui)
# ---------------------------------------------------------------------------
add_library(syntraq_renderer STATIC
    src/rendering/renderer.cpp
)
target_include_directories(syntraq_renderer PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(syntraq_renderer PUBLIC syntraq_core raylib rlimgui imgui)
target_compile_options(syntraq_renderer PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /EHsc>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

# ---------------------------------------------------------------------------
# syntraq — main executable
# ---------------------------------------------------------------------------
add_executable(syntraq src/main.cpp)
target_link_libraries(syntraq PRIVATE syntraq_core syntraq_renderer)
target_compile_options(syntraq PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /EHsc>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

# On Windows, copy raylib.dll next to the executable after build
if(WIN32 AND EXISTS "${RAYLIB_SDK_DIR}/lib/raylib.dll")
    add_custom_command(TARGET syntraq POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RAYLIB_SDK_DIR}/lib/raylib.dll"
            "$<TARGET_FILE_DIR:syntraq>/raylib.dll"
        COMMENT "Copying raylib.dll to output directory"
    )
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
add_executable(syntraq_benchmarks benchmarks/bench_routing.cpp)
target_link_libraries(syntraq_benchmarks PRIVATE syntraq_core)
target_compile_options(syntraq_benchmarks PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /EHsc>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

# ---------------------------------------------------------------------------
# Experiments Runner
# ---------------------------------------------------------------------------
add_executable(syntraq_experiments experiments/run_experiments.cpp)
target_link_libraries(syntraq_experiments PRIVATE syntraq_core)
target_compile_options(syntraq_experiments PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /EHsc>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# ---------------------------------------------------------------------------
# Install (basic)
# ---------------------------------------------------------------------------
install(TARGETS syntraq DESTINATION bin)
if(WIN32 AND EXISTS "${RAYLIB_SDK_DIR}/lib/raylib.dll")
    install(FILES "${RAYLIB_SDK_DIR}/lib/raylib.dll" DESTINATION bin)
endif()
