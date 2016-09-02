# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)

project( TerrainGenerator )

cmake_minimum_required( VERSION 2.8 )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules )

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules ${CMAKE_MODULE_PATH} )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-deprecated-declarations -Wno-unused-private-field -Wno-unused-value -Wno-overloaded-virtual -Wmissing-braces" )

find_package( Pangolin REQUIRED )

set( _INCLUDE_DIRS
        ${Pangolin_INCLUDE_DIRS}
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

set( _LIBRARIES
        ${Pangolin_LIBRARIES}
        )

include_directories( ${_INCLUDE_DIRS} )

set( HDRS
        include/TerrainGenerator/Generator.h
        )

set( SRCS
        src/Generator.cpp
        )

set( LIBRARY_NAME ${PROJECT_NAME} )
string( TOLOWER ${LIBRARY_NAME} LIBRARY_NAME )
add_library( ${LIBRARY_NAME} ${SRCS} ${HDRS} )
target_link_libraries( ${LIBRARY_NAME} ${_LIBRARIES} )

add_executable( generator src/Generator.cpp )
target_link_libraries( generator ${LIBRARY_NAME} )
