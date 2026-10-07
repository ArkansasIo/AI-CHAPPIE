cmake_minimum_required(VERSION 3.16)
project(ChappieClone VERSION 1.0.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++17, warnings on, no compiler-specific extensions.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Core engine library -- all subsystems live here.
# ---------------------------------------------------------------------------
add_library(chappie_core STATIC
    src/core/math.cpp
    src/core/json.cpp
    src/core/config.cpp
    src/core/logger.cpp

    src/ai/neural_network.cpp
    src/ai/brain.cpp
    src/ai/memory.cpp
    src/ai/personality.cpp
    src/ai/decision_engine.cpp

    src/vision/vision.cpp
    src/speech/speech.cpp
    src/robotics/robot_body.cpp
    src/navigation/navigator.cpp
    src/sensors/sensors.cpp
    src/networking/networking.cpp
    src/security/security.cpp

    src/database/database.cpp

    src/plugins/plugin_manager.cpp
    src/api_server.cpp
)

target_include_directories(chappie_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# SQLite: use the system copy if present, otherwise the bundled amalgamation.
find_path(SQLITE3_INCLUDE_DIR sqlite3.h)
find_library(SQLITE3_LIBRARY sqlite3)
if(SQLITE3_INCLUDE_DIR AND SQLITE3_LIBRARY)
    message(STATUS "Using system SQLite: ${SQLITE3_LIBRARY}")
    target_include_directories(chappie_core PUBLIC ${SQLITE3_INCLUDE_DIR})
    target_link_libraries(chappie_core PUBLIC ${SQLITE3_LIBRARY})
else()
    message(STATUS "System SQLite not found -- using bundled sqlite3 amalgamation")
    target_sources(chappie_core PRIVATE third_party/sqlite3/sqlite3.c)
    target_include_directories(chappie_core PUBLIC third_party/sqlite3)
endif()

if(WIN32)
    target_link_libraries(chappie_core PUBLIC ws2_32)
endif()

# ---------------------------------------------------------------------------
# Main executable -- the robot runtime / headless simulator.
# ---------------------------------------------------------------------------
add_executable(chappie src/main.cpp)
target_link_libraries(chappie PRIVATE chappie_core)

# ---------------------------------------------------------------------------
# Tests (simple assert-based, no external framework required).
# ---------------------------------------------------------------------------
enable_testing()
add_executable(chappie_tests
    tests/test_main.cpp
    tests/test_math.cpp
    tests/test_neural_network.cpp
    tests/test_memory.cpp
    tests/test_navigation.cpp
    tests/test_security.cpp
    tests/test_vision.cpp
)
target_link_libraries(chappie_tests PRIVATE chappie_core)
add_test(NAME chappie_tests COMMAND chappie_tests)

message(STATUS "ChappieClone ${PROJECT_VERSION} configured (${CMAKE_BUILD_TYPE})")
