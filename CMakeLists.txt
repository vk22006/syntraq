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
# Raylib — use the pre-built local SDK (ARM64 MSVC build)
# ---------------------------------------------------------------------------
set(RAYLIB_SDK_DIR "D:/Game development projects/raylib-6.0_winarm64_msvc16"
    CACHE PATH "Path to the pre-built Raylib SDK")

# Import raylib as an interface target so other targets can link against it
add_library(raylib SHARED IMPORTED GLOBAL)
set_target_properties(raylib PROPERTIES
    IMPORTED_IMPLIB   "${RAYLIB_SDK_DIR}/lib/raylibdll.lib"
    IMPORTED_LOCATION "${RAYLIB_SDK_DIR}/lib/raylib.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${RAYLIB_SDK_DIR}/include"
)

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
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR})
# Suppress MSVC warnings in third-party imgui source
target_compile_options(imgui PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W0>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>
)

# ---------------------------------------------------------------------------
# rlImGui — compile from the local SDK copy
# ---------------------------------------------------------------------------
set(RLIMGUI_DIR "${RAYLIB_SDK_DIR}/rlImGui-Raylib_6_0")

add_library(rlimgui STATIC
    "${RLIMGUI_DIR}/rlImGui.cpp"
)
target_include_directories(rlimgui PUBLIC
    "${RLIMGUI_DIR}"
    "${RAYLIB_SDK_DIR}/include"   # raylib.h, rlgl.h
    ${imgui_SOURCE_DIR}           # imgui.h
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
)
target_include_directories(syntraq_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
# Warning flags for our own code
target_compile_options(syntraq_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive->
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
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive->
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

# ---------------------------------------------------------------------------
# syntraq — main executable
# ---------------------------------------------------------------------------
add_executable(syntraq src/main.cpp)
target_link_libraries(syntraq PRIVATE syntraq_core syntraq_renderer)
target_compile_options(syntraq PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive->
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

# On Windows, copy raylib.dll next to the executable after build
add_custom_command(TARGET syntraq POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${RAYLIB_SDK_DIR}/lib/raylib.dll"
        "$<TARGET_FILE_DIR:syntraq>/raylib.dll"
    COMMENT "Copying raylib.dll to output directory"
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
install(FILES "${RAYLIB_SDK_DIR}/lib/raylib.dll" DESTINATION bin)
