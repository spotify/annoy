cmake_minimum_required(VERSION 3.14)
project(Annoy
    DESCRIPTION "Approximate Nearest Neighbors Oh Yeah"
    VERSION 1.17.0
    LANGUAGES CXX
)

add_library(Annoy INTERFACE)

set(ANNOY_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/annoy)
file(MAKE_DIRECTORY ${ANNOY_INCLUDE_DIR})
foreach (HEADER annoylib.h kissrandom.h mman.h)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/${HEADER} DESTINATION ${ANNOY_INCLUDE_DIR})
endforeach()

target_include_directories(Annoy INTERFACE include/)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Add Python set-up code here.
endif()

