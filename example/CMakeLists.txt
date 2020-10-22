cmake_minimum_required(VERSION 3.5.1)
project(apriltag_pose_estimation)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(APRILTAG_DIR /usr/local/include/apriltag)
set(LIB_APRILTAG_DIR /usr/local/lib)

include_directories(${APRILTAG_DIR})
link_directories(${LIB_APRILTAG_DIR})

add_executable(apriltag_pose_estimation
        apriltag_pose_estimation.cc)

target_link_libraries(apriltag_pose_estimation ${OpenCV_LIBS} libapriltag.so.3)

