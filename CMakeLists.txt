cmake_minimum_required(VERSION 3.0.0)
project(FAW-Volkswagen VERSION 0.1.0)

include(CTest)
enable_testing()

add_subdirectory(${PROJECT_SOURCE_DIR}/cv/)

add_executable(FAW-Volkswagen main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
