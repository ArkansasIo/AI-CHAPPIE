cmake_minimum_required(VERSION 3.16)
project(AIChappie LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chappie_core
  src/brain/BrainOS.cpp
  src/boot/BootManager.cpp
  src/safety/LawEngine.cpp
  src/safety/SafetySupervisor.cpp
  src/model/RobotModel.cpp
  src/hardware/SimulatedHardware.cpp
)
target_include_directories(chappie_core PUBLIC src)
if(MSVC)
  target_compile_options(chappie_core PRIVATE /W4)
else()
  target_compile_options(chappie_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(chappie_demo apps/chappie_demo.cpp)
target_link_libraries(chappie_demo PRIVATE chappie_core)

enable_testing()
add_executable(chappie_tests tests/chappie_tests.cpp)
target_link_libraries(chappie_tests PRIVATE chappie_core)
add_test(NAME chappie_tests COMMAND chappie_tests)
