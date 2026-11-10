cmake_minimum_required(VERSION 3.27)

if(NOT TARGET tiny-metal-nn)
    project(tinymetalnn_mlx LANGUAGES CXX)
    set(METALNN_BUILD_MLX ON CACHE BOOL "Build MLX" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/_native/CMakeLists.txt")
        set(native_root "${CMAKE_CURRENT_SOURCE_DIR}/_native")
    else()
        get_filename_component(native_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    endif()
    add_subdirectory("${native_root}" native)
    return()
endif()

enable_language(OBJCXX)
execute_process(COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_VARIABLE mlx_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
find_package(MLX 0.32.4 EXACT CONFIG REQUIRED HINTS "${mlx_cmake_dir}")
if(NOT MLX_BUILD_METAL)
    message(FATAL_ERROR "The installed MLX must have Metal support")
endif()
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind 3.1.0 EXACT CONFIG REQUIRED PATHS "${nanobind_cmake_dir}" NO_DEFAULT_PATH)

add_library(tinymetalnn_mlx_native STATIC src/module.cpp src/mlx_bridge.mm)
target_compile_features(tinymetalnn_mlx_native PUBLIC cxx_std_20)
target_include_directories(tinymetalnn_mlx_native PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" ${MLX_INCLUDE_DIRS})
target_link_libraries(tinymetalnn_mlx_native PUBLIC mlx tiny-metal-nn::tiny-metal-nn)
target_compile_options(tinymetalnn_mlx_native PRIVATE -fno-strict-aliasing
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc;-fblocks>")
set_target_properties(tinymetalnn_mlx_native PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden OBJCXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED YES)

nanobind_add_module(tinymetalnn_mlx NB_STATIC NB_DOMAIN mlx bindings.cpp)
target_link_libraries(tinymetalnn_mlx PRIVATE tinymetalnn_mlx_native)
set_target_properties(tinymetalnn_mlx PROPERTIES OUTPUT_NAME _C
    LIBRARY_OUTPUT_DIRECTORY "${METALNN_PYTHON_OUTPUT_DIRECTORY}/tinymetalnn_mlx")
file(MAKE_DIRECTORY "${METALNN_PYTHON_OUTPUT_DIRECTORY}/tinymetalnn_mlx")
file(GLOB python_files CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tinymetalnn_mlx/*.py")
foreach(source IN LISTS python_files)
    get_filename_component(filename "${source}" NAME)
    configure_file("${source}" "${METALNN_PYTHON_OUTPUT_DIRECTORY}/tinymetalnn_mlx/${filename}" COPYONLY)
endforeach()
file(COPY "${PROJECT_SOURCE_DIR}/kernels/"
    DESTINATION "${METALNN_PYTHON_OUTPUT_DIRECTORY}/tinymetalnn_mlx/rtc/include")
