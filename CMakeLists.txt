cmake_minimum_required(VERSION 3.27)
project(tiny_metal_nn VERSION 0.6.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(METALNN_BUILD_TORCH "Build the independent PyTorch MPS package" OFF)
option(METALNN_BUILD_MLX "Build the independent MLX Metal package" OFF)
option(TCNN_HALF_PRECISION "Use FP16 network parameters by default" ON)
option(TCNN_PARAMS_UNALIGNED "Support framework-owned parameter buffers without alignment guarantees" ON)
option(TCNN_NO_NETWORKS "Build encoding factories without network factories" OFF)
option(TCNN_BUILD_WITH_RTC "Enable runtime Metal compilation and model fusion" ON)
option(TCNN_BUILD_USE_FAST_MATH "Allow relaxed arithmetic in generated Metal kernels" OFF)
option(TCNN_BUILD_NO_FWD_BWD "Build JIT-only forward and backward interfaces" OFF)

if(TCNN_BUILD_NO_FWD_BWD AND NOT TCNN_BUILD_WITH_RTC)
    message(FATAL_ERROR "TCNN_BUILD_NO_FWD_BWD requires TCNN_BUILD_WITH_RTC")
endif()
if(METALNN_BUILD_TORCH OR METALNN_BUILD_MLX)
    if(NOT APPLE)
        message(FATAL_ERROR "The Python bindings require macOS and native Metal support")
    endif()
    if(NOT TCNN_PARAMS_UNALIGNED)
        message(FATAL_ERROR "Python bindings require TCNN_PARAMS_UNALIGNED=ON")
    endif()
endif()
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

foreach(required_file metalrt/CMakeLists.txt fmt/CMakeLists.txt nlohmann/json.hpp pcg32/pcg32.h)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/${required_file}")
        message(FATAL_ERROR "Missing dependencies/${required_file}. Run git submodule update --init --recursive")
    endif()
endforeach()
if(NOT TARGET metalrt::metalrt)
    set(METALRT_BUILD_TESTS OFF CACHE BOOL "Build MetalRT tests")
    set(METALRT_BUILD_EXAMPLES OFF CACHE BOOL "Build MetalRT examples")
    add_subdirectory(dependencies/metalrt)
    set_target_properties(metalrt PROPERTIES CXX_VISIBILITY_PRESET hidden
        OBJCXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()
if((METALNN_BUILD_TORCH OR METALNN_BUILD_MLX) AND NOT METALRT_ENABLE_METAL)
    message(FATAL_ERROR "Python bindings require METALRT_ENABLE_METAL=ON")
endif()
if(NOT TARGET fmt::fmt)
    set(FMT_INSTALL ON CACHE BOOL "Install fmt for C++ consumers")
    set(FMT_TEST OFF CACHE BOOL "Build fmt tests")
    add_subdirectory(dependencies/fmt)
    set_target_properties(fmt PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()

# Embed handwritten assets through CMake itself. No generator script or Python is
# used, and installed binaries retain their shaders after the source tree moves.
file(GLOB_RECURSE METALNN_SHADER_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.metal")
list(SORT METALNN_SHADER_FILES)
set(METALNN_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${METALNN_GENERATED_DIR}")
set(METALNN_EMBEDDED_HEADER "#pragma once\n#include <map>\n#include <string>\nnamespace tcnn { namespace detail {\ninline const std::map<std::string,std::string>& embedded_nn_shaders() {\nstatic const std::map<std::string,std::string> shaders = {\n")
foreach(METALNN_SHADER_FILE IN LISTS METALNN_SHADER_FILES)
    file(RELATIVE_PATH METALNN_SHADER_NAME "${CMAKE_CURRENT_SOURCE_DIR}/kernels" "${METALNN_SHADER_FILE}")
    file(READ "${METALNN_SHADER_FILE}" METALNN_SHADER_TEXT)
    string(APPEND METALNN_EMBEDDED_HEADER "{\"${METALNN_SHADER_NAME}\",R\"TCNN_MSL(${METALNN_SHADER_TEXT})TCNN_MSL\"},\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${METALNN_SHADER_FILE}")
endforeach()
string(APPEND METALNN_EMBEDDED_HEADER "};\nreturn shaders;\n}\n}}\n")
file(WRITE "${METALNN_GENERATED_DIR}/nn_shaders_generated.h" "${METALNN_EMBEDDED_HEADER}")
set(METALNN_SOURCE_FILES
    src/cpp_api.cpp
    src/encoding_jit.cpp
    src/encodings.cpp
    src/foundation_kernels.cpp
    src/fused_mlp.cpp
    src/jit.cpp
    src/jit_gradients.cpp
    src/jit_network.cpp
    src/losses.cpp
    src/metal_runtime.cpp
    src/nn.cpp
    src/optimizers.cpp
    src/reduce_sum.cpp
    src/rtc_kernel.cpp
    src/shaders.cpp)
add_library(tiny-metal-nn STATIC ${METALNN_SOURCE_FILES})
add_library(tiny-metal-nn::tiny-metal-nn ALIAS tiny-metal-nn)
target_compile_features(tiny-metal-nn PUBLIC cxx_std_17)
target_link_libraries(tiny-metal-nn PUBLIC metalrt::metalrt fmt::fmt)
target_include_directories(tiny-metal-nn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dependencies>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE "${METALNN_GENERATED_DIR}")
target_compile_definitions(tiny-metal-nn PUBLIC
    TCNN_HALF_PRECISION=$<BOOL:${TCNN_HALF_PRECISION}> FMT_UNICODE=0
    $<$<BOOL:${TCNN_PARAMS_UNALIGNED}>:TCNN_PARAMS_UNALIGNED=1>
    $<$<BOOL:${TCNN_NO_NETWORKS}>:TCNN_NO_NETWORKS=1>
    $<$<BOOL:${TCNN_BUILD_WITH_RTC}>:TCNN_RTC=1>
    $<$<BOOL:${TCNN_BUILD_USE_FAST_MATH}>:TCNN_RTC_USE_FAST_MATH=1>
    $<$<BOOL:${TCNN_BUILD_NO_FWD_BWD}>:TCNN_NO_FWD_BWD=1>)
set_target_properties(tiny-metal-nn PROPERTIES
    CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tiny-metal-nn PRIVATE -Wall -Wextra -Wpedantic)
endif()

# The default build is C++ only: Python and frameworks are discovered on demand.
if(METALNN_BUILD_TORCH OR METALNN_BUILD_MLX)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
    set(METALNN_PYTHON_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python" CACHE PATH
        "Importable Python packages produced by CMake")
endif()
if(METALNN_BUILD_TORCH)
    add_subdirectory(bindings/torch)
endif()
if(METALNN_BUILD_MLX)
    add_subdirectory(bindings/mlx)
endif()

install(TARGETS tiny-metal-nn EXPORT tiny-metal-nn-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY dependencies/nlohmann dependencies/pcg32
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(DIRECTORY kernels/ DESTINATION ${CMAKE_INSTALL_DATADIR}/tiny-metal-nn/kernels)
install(EXPORT tiny-metal-nn-targets NAMESPACE tiny-metal-nn::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tiny-metal-nn)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tiny-metal-nn-config.cmake"
    "include(CMakeFindDependencyMacro)\nfind_dependency(metalrt 0.3 CONFIG)\nfind_dependency(fmt 11.2 CONFIG)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/tiny-metal-nn-targets.cmake\")\n")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/tiny-metal-nn-config-version.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tiny-metal-nn-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tiny-metal-nn-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tiny-metal-nn)

# Only format project-owned host sources. Dependencies and build products are excluded.
find_program(METALNN_CLANG_FORMAT NAMES clang-format)
if(METALNN_CLANG_FORMAT)
    file(GLOB_RECURSE METALNN_FORMAT_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/torch/tinymetalnn/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/torch/tinymetalnn/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/torch/tinymetalnn/*.mm"
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/mlx/src/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/mlx/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/mlx/src/*.mm")
    list(APPEND METALNN_FORMAT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/bindings/mlx/bindings.cpp")
    add_custom_target(format COMMAND "${METALNN_CLANG_FORMAT}" -i ${METALNN_FORMAT_FILES}
        COMMAND_EXPAND_LISTS VERBATIM COMMENT "Formatting project C++ and Objective-C++ sources")
    add_custom_target(format-check COMMAND "${METALNN_CLANG_FORMAT}" --dry-run --Werror ${METALNN_FORMAT_FILES}
        COMMAND_EXPAND_LISTS VERBATIM COMMENT "Checking project source formatting")
endif()

# Native training regressions are opt-in; no Python bindings or fixtures required.
option(METALNN_BUILD_TESTS "Build native Metal neural-network regression tests" OFF)
if(METALNN_BUILD_TESTS)
    if(NOT METALRT_ENABLE_METAL OR TCNN_NO_NETWORKS OR TCNN_BUILD_NO_FWD_BWD)
        message(FATAL_ERROR "Native training tests require Metal, networks and AOT forward/backward")
    endif()
    enable_testing()
    add_executable(bench_encoder_training tests/bench_encoder_training.cpp)
    target_link_libraries(bench_encoder_training PRIVATE tiny-metal-nn::tiny-metal-nn)
    add_executable(bench_analytic_encodings tests/bench_analytic_encodings.cpp)
    target_link_libraries(bench_analytic_encodings PRIVATE tiny-metal-nn::tiny-metal-nn)
    add_executable(test_hashgrid_3d tests/hashgrid_3d.cpp)
    target_link_libraries(test_hashgrid_3d PRIVATE tiny-metal-nn::tiny-metal-nn)
    add_test(NAME hashgrid_3d COMMAND test_hashgrid_3d)
    set_tests_properties(hashgrid_3d PROPERTIES RUN_SERIAL TRUE)
    add_executable(test_analytic_encodings tests/analytic_encodings.cpp)
    target_link_libraries(test_analytic_encodings PRIVATE tiny-metal-nn::tiny-metal-nn)
    add_test(NAME analytic_encodings COMMAND test_analytic_encodings)
    set_tests_properties(analytic_encodings PROPERTIES RUN_SERIAL TRUE)
    foreach(precision IN ITEMS fp16 fp32)
        add_executable(test_wide_training_${precision} tests/wide_training.cpp)
        target_link_libraries(test_wide_training_${precision} PRIVATE tiny-metal-nn::tiny-metal-nn)
        target_compile_features(test_wide_training_${precision} PRIVATE cxx_std_17)
        if(precision STREQUAL "fp32")
            target_compile_definitions(test_wide_training_${precision} PRIVATE WIDE_TEST_FP32=1)
        endif()
        add_test(NAME wide_training_${precision} COMMAND test_wide_training_${precision})
        set_tests_properties(wide_training_${precision} PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endforeach()
endif()
