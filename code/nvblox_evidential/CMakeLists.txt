# nvblox_evidential — CMake build.
cmake_minimum_required(VERSION 3.22)
project(nvblox_evidential LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    # 86 = RTX 4060 (Ada Lovelace lower bin), 87 = Jetson Orin NX.
    set(CMAKE_CUDA_ARCHITECTURES "86;87")
endif()

find_package(CUDAToolkit REQUIRED)

# Upstream nvblox is consumed as an external target. We expect it to be either
# (a) installed system-wide via the Isaac ROS release, or (b) supplied as a
# CMake subdirectory by the parent project.
find_package(nvblox QUIET)
if(NOT TARGET nvblox::nvblox_lib)
    message(STATUS
        "[nvblox_evidential] upstream nvblox not found via find_package; "
        "set NVBLOX_DIR or vendor it under external/nvblox before building."
    )
endif()

# Optional: Python bindings via pybind11. Skip if pybind11 is not installed
# so that pure-C++ users can still build the layer.
find_package(pybind11 QUIET)

# ----------------------------------------------------------------------------
# Static library — the C++ side of the EvidentialLayer.
# ----------------------------------------------------------------------------
add_library(nvblox_evidential_core STATIC
    src/evidential_layer.cu
    src/evidential_integrator.cu
)
target_include_directories(nvblox_evidential_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(nvblox_evidential_core PUBLIC cxx_std_20)
target_link_libraries(nvblox_evidential_core
    PUBLIC
        CUDA::cudart
        $<TARGET_NAME_IF_EXISTS:nvblox::nvblox_lib>
)
set_target_properties(nvblox_evidential_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)

# ----------------------------------------------------------------------------
# Optional Python bindings.
# ----------------------------------------------------------------------------
if(pybind11_FOUND)
    pybind11_add_module(_nvblox_evidential MODULE
        src/python_bindings.cpp
    )
    target_link_libraries(_nvblox_evidential PRIVATE nvblox_evidential_core)
    set_target_properties(_nvblox_evidential PROPERTIES
        LIBRARY_OUTPUT_NAME "nvblox_evidential"
    )
endif()

# ----------------------------------------------------------------------------
# Tests (GTest) — skipped if GTest is unavailable.
# ----------------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        enable_testing()
        add_executable(test_evidential_layer tests/test_evidential_layer.cpp)
        target_link_libraries(test_evidential_layer
            PRIVATE
                nvblox_evidential_core
                GTest::gtest_main
        )
        add_test(NAME evidential_layer_basic COMMAND test_evidential_layer)
    else()
        message(STATUS
            "[nvblox_evidential] GTest not found; C++ tests will not be built."
        )
    endif()
endif()
