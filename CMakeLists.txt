cmake_minimum_required(VERSION 3.15...3.25 FATAL_ERROR)

project(Annoy
  DESCRIPTION "Approximate Nearest Neighbors Oh Yeah"
  VERSION 1.17.1
  LANGUAGES CXX)

add_library(Annoy INTERFACE)
add_library(Annoy::Annoy ALIAS Annoy)

foreach (HEADER annoylib.h kissrandom.h mman.h)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/${HEADER}" "${CMAKE_CURRENT_BINARY_DIR}/include/annoy/${HEADER}" COPYONLY)
endforeach ()

target_include_directories(Annoy INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Install
include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS Annoy
  EXPORT AnnoyTargets)

install(EXPORT AnnoyTargets
  FILE AnnoyConfig.cmake
  NAMESPACE Annoy::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/annoy)

export(TARGETS Annoy NAMESPACE Annoy:: FILE AnnoyConfig.cmake)
