cmake_minimum_required(VERSION 3.27)

# This directory also works as a standalone CMake entry point.
if(NOT TARGET tiny-metal-nn)
    project(tinymetalnn_torch LANGUAGES CXX)
    set(METALNN_BUILD_TORCH ON CACHE BOOL "Build PyTorch" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/_native/CMakeLists.txt")
        set(native_root "${CMAKE_CURRENT_SOURCE_DIR}/_native")
    else()
        get_filename_component(native_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    endif()
    add_subdirectory("${native_root}" native)
    return()
endif()

enable_language(OBJCXX)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE torch_cmake_prefix OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE torch_result)
if(NOT torch_result EQUAL 0)
    message(FATAL_ERROR "Install PyTorch >=2.7 in ${Python_EXECUTABLE}")
endif()
find_package(Torch 2.7 CONFIG REQUIRED PATHS "${torch_cmake_prefix}/Torch" NO_DEFAULT_PATH)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
# Use the pybind11 headers supplied by this Torch installation: its type casters
# must match Torch's own pybind11 ABI, even when Torch changes major versions.
Python_add_library(tinymetalnn_torch MODULE WITH_SOABI
    tinymetalnn/bindings.cpp tinymetalnn/mps_bridge.mm)
target_compile_features(tinymetalnn_torch PRIVATE cxx_std_20)
target_compile_definitions(tinymetalnn_torch PRIVATE
    TORCH_EXTENSION_NAME=_C TORCH_API_INCLUDE_EXTENSION_H)
# Older Torch releases publish the ABI tags used by their bundled pybind11.
foreach(abi COMPILER_TYPE STDLIB BUILD_ABI)
    execute_process(COMMAND "${Python_EXECUTABLE}" -c
        "import torch; print(getattr(torch._C, '_PYBIND11_${abi}', ''))"
        OUTPUT_VARIABLE abi_value OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    if(abi_value)
        target_compile_definitions(tinymetalnn_torch PRIVATE "PYBIND11_${abi}=\"${abi_value}\"")
    endif()
endforeach()
target_include_directories(tinymetalnn_torch SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(tinymetalnn_torch PRIVATE
    "${PROJECT_SOURCE_DIR}/dependencies/pybind11_json/include")
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(tinymetalnn_torch PRIVATE ${torch_flags} -fno-strict-aliasing
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc;-fblocks>")
target_link_libraries(tinymetalnn_torch PRIVATE tiny-metal-nn::tiny-metal-nn
    ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}"
    "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph")
set_target_properties(tinymetalnn_torch PROPERTIES OUTPUT_NAME _C
    CXX_VISIBILITY_PRESET hidden OBJCXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED YES
    LIBRARY_OUTPUT_DIRECTORY "${METALNN_PYTHON_OUTPUT_DIRECTORY}/tinymetalnn_bindings"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
foreach(package tinymetalnn tinymetalnn_bindings)
    file(MAKE_DIRECTORY "${METALNN_PYTHON_OUTPUT_DIRECTORY}/${package}")
    file(GLOB python_files CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${package}/*.py")
    foreach(source IN LISTS python_files)
        get_filename_component(filename "${source}" NAME)
        configure_file("${source}" "${METALNN_PYTHON_OUTPUT_DIRECTORY}/${package}/${filename}" COPYONLY)
    endforeach()
endforeach()
file(COPY "${PROJECT_SOURCE_DIR}/kernels/"
    DESTINATION "${METALNN_PYTHON_OUTPUT_DIRECTORY}/tinymetalnn/rtc/include")
