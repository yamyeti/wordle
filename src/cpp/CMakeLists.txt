cmake_minimum_required(VERSION 3.1...3.29)

project(
  wordle
  VERSION 1.0
  LANGUAGES CXX
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(wordle wordle.cpp)

add_subdirectory(game)

target_link_libraries(wordle PUBLIC game)

target_include_directories(
  wordle PUBLIC
  "${PROJECT_BINARY_DIR}"
  "${PROJECT_SOURCE_DIR}/game"
)