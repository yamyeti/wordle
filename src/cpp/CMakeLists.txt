cmake_minimum_required(VERSION 3.1...3.29)

project(
  wordle
  VERSION 1.0
  LANGUAGES CXX
)

add_executable(wordle wordle.cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)