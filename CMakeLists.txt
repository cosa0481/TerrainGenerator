# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)

project( TerrainGenerator )

cmake_minimum_required( VERSION 2.8 )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules )

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules ${CMAKE_MODULE_PATH} )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-deprecated-declarations -Wno-unused-private-field -Wno-unused-value -Wno-overloaded-virtual -Wmissing-braces" )

find_package( Pangolin REQUIRED )
link_directories( ${Pangolin_LIBRARY_DIRS} )
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable( TerrainGenerator main.cpp )
target_link_libraries( TerrainGenerator ${Pangolin_LIBRARIES} )
