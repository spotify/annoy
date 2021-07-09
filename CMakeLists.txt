project(Annoy
    DESCRIPTION "Approximate Nearest Neighbors Oh Yeah"
    VERSION 1.17.0
    LANGUAGES CXX
)
cmake_minimum_required(VERSION 3.14)

add_library(Annoy INTERFACE)

target_include_directories(Annoy INTERFACE src/)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Add Python set-up code here.
endif()

