cmake_minimum_required(VERSION 2.8)
PROJECT(FeatureSearch)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SKIP_RPATH ON)
set(CMAKE_MACOSX_RPATH 0)
if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

include_directories (
    ${PROJECT_SOURCE_DIR}/../src
)

ADD_EXECUTABLE(
    precision_test
    precision_test.cpp
    ../src/annoylib.h
)

TARGET_LINK_LIBRARIES(
    precision_test
    ${OpenCV_LIBS}
)