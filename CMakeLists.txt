cmake_minimum_required(VERSION 3.20)
project(simplical_spectral_bandit_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

option(SB_BUILD_VIEWER "Build interactive viewers (raylib)" ON)
option(SB_BUILD_TESTS "Build C++ tests" ON)

include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()

FetchContent_Declare(
  spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG v1.1.0
)
FetchContent_GetProperties(spectra)
if(NOT spectra_POPULATED)
  FetchContent_Populate(spectra)
endif()

if(SB_BUILD_VIEWER)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(USE_AUDIO OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.5
  )
  FetchContent_MakeAvailable(raylib)
endif()

add_library(sb_core
  src/mesh.cpp
  src/geometry.cpp
  src/heat.cpp
  src/patchification.cpp
  src/spectral.cpp
  src/bandit.cpp
  src/pathfinding.cpp
  src/simulator.cpp
)

target_include_directories(sb_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
    ${spectra_SOURCE_DIR}/include
)

if(APPLE)
  target_compile_definitions(sb_core PRIVATE _USE_MATH_DEFINES)
endif()

if(SB_BUILD_VIEWER)
  add_library(sb_viewer
    src/viewer.cpp
  )
  target_include_directories(sb_viewer
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${eigen_SOURCE_DIR}
      ${spectra_SOURCE_DIR}/include
  )
  target_link_libraries(sb_viewer PUBLIC sb_core raylib)
endif()

add_executable(spectral_bandit_cpp
  src/main.cpp
)

target_link_libraries(spectral_bandit_cpp PRIVATE sb_core)
if(SB_BUILD_VIEWER)
  target_link_libraries(spectral_bandit_cpp PRIVATE sb_viewer)
  target_compile_definitions(spectral_bandit_cpp PRIVATE SB_WITH_VIEWER=1)
endif()

if(SB_BUILD_TESTS)
  add_executable(sb_tests
    tests_cpp/test_pipeline.cpp
  )
  target_link_libraries(sb_tests PRIVATE sb_core)
  if(SB_BUILD_VIEWER)
    target_link_libraries(sb_tests PRIVATE sb_viewer)
    target_compile_definitions(sb_tests PRIVATE SB_WITH_VIEWER=1)
  endif()
  target_compile_definitions(sb_tests PRIVATE SB_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME sb_tests COMMAND sb_tests)
endif()
