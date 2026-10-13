cmake_minimum_required(VERSION 3.16)
project(ChappieClone VERSION 1.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Preserve existing core, neural-network, memory, and audio implementations.
add_library(chappie_core STATIC
  src/core/math.cpp src/core/json.cpp src/core/config.cpp src/core/logger.cpp
  src/ai/neural_network.cpp src/ai/memory.cpp
  src/audio/audio.cpp src/audio/phoneme.cpp src/audio/tts.cpp
  src/robot_runtime.cpp)
target_include_directories(chappie_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(WIN32)
  target_link_libraries(chappie_core PUBLIC ws2_32)
endif()
add_executable(chappie src/main.cpp)
target_link_libraries(chappie PRIVATE chappie_core)
enable_testing()
add_executable(chappie_tests tests/test_runtime.cpp)
target_link_libraries(chappie_tests PRIVATE chappie_core)
add_test(NAME runtime_tests COMMAND chappie_tests)
message(STATUS "ChappieClone ${PROJECT_VERSION} configured (${CMAKE_BUILD_TYPE})")
