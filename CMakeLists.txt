cmake_minimum_required(VERSION 3.10)
set(Basalt_DIR "$ENV{HOME}/basalt")

project(dynamics_vio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Flags used for CHECK_CXX_SOURCE_COMPILES
set(CMAKE_REQUIRED_FLAGS "-Wno-error")

set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")          # cmake default: "-g"
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")  # cmake default: "-O2 -g -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")                                      # cmake default: "-O3 -DNDEBUG"
set(CMAKE_CXX_FLAGS_CIDEBUG  "-O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")          # CI version with no debug symbols
set(CMAKE_CXX_FLAGS_CIRELWITHDEBINFO "-O3 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")  # CI version with no debug symbols

# base set of compile flags
set(BASALT_CXX_FLAGS "-Wall -Wextra -Werror -Wno-error=unused-variable -Wno-error=ignored-attributes -Wno-error=array-bounds -Wno-error=sign-compare  -ftemplate-backtrace-limit=0")

set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")

if(NOT CXX_MARCH)
  set(CXX_MARCH native)
endif()

set(BASALT_PASSED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${BASALT_CXX_FLAGS} -march=${CXX_MARCH} ${BASALT_PASSED_CXX_FLAGS}")

set(CMAKE_MODULE_PATH "${Basalt_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

set(USER_INC include)

if(NOT EIGEN_ROOT)
  set(EIGEN_ROOT "${Basalt_DIR}/thirdparty/basalt-headers/thirdparty/eigen")
endif()
set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.3.7 EXACT REQUIRED MODULE)
list(APPEND USER_INC  ${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")

find_package(TBB REQUIRED)
list(APPEND USER_INC  ${TBB_INCLUDE_DIR})

message(STATUS "TBB_INTERFACE_VERSION: ${TBB_INTERFACE_VERSION}")
if (TBB_INTERFACE_VERSION LESS 11004)
  # enable global_control header for earlier TBB versions (Ubuntu 16.04, 18.04)
  add_definitions(-DTBB_PREVIEW_GLOBAL_CONTROL)
endif()

find_package(Pangolin REQUIRED PATHS ${Basalt_DIR}/build/thirdparty/Pangolin/src  NO_DEFAULT_PATH)
list(APPEND USER_INC  ${Pangolin_INCLUDE_DIR})

find_library(ROSBAG_LIBRARY
    NAMES rosbag
    HINTS "${Basalt_DIR}/build/thirdparty/ros"
)

list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/ros/include)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/ros/console_bridge/include)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/ros/roscpp_core/cpp_common/include)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/ros/roscpp_core/rostime/include)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/ros/roscpp_core/roscpp_serialization/include)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/ros/roscpp_core/roscpp_traits/include)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/ros/ros_comm/utilities/roslz4/include)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/ros/ros_comm/tools/rosbag_storage/include)
list(APPEND USER_INC  $ENV{ACADO_ENV_INCLUDE_DIRS})
message(STATUS "ACADO_ENV_INCLUDE_DIRS: $ENV{ACADO_ENV_INCLUDE_DIRS}")

add_library(basalt SHARED IMPORTED)
set_target_properties(basalt PROPERTIES IMPORTED_LOCATION ${Basalt_DIR}/build/libbasalt.so)

list(APPEND USER_INC  thirdparty/ros/include)
list(APPEND USER_INC  ${Basalt_DIR}/include)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/basalt-headers/thirdparty/Sophus)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/basalt-headers/thirdparty/cereal/include)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/basalt-headers/include)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/ros)
list(APPEND USER_INC  ${Basalt_DIR}/thirdparty/CLI11/include)

link_directories($ENV{ACADO_ENV_LIBRARY_DIRS})

add_library(${PROJECT_NAME} SHARED
    src/vio_estimator/vio_base.cpp
    src/single_track/single_track_factor.cpp
    src/single_track/single_track_acado.cpp
    src/single_track/constraints_factor.cpp
    src/single_track/single_track_linearizer.cpp
    src/single_track/constraints_factor_linearizer.cpp
    src/io/dyn_vio_io.cpp
    src/utils/utils.cpp
    src/parameters/parameters.cpp
        src/forward/single_track_forward.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${USER_INC})
target_link_libraries(${PROJECT_NAME} PUBLIC stdc++fs basalt ${TBB_LIBRARIES} $ENV{ACADO_ENV_SHARED_LIBRARIES})

add_executable(app_dynamics_vio "src/app_dynamics_vio.cpp")
target_link_libraries(app_dynamics_vio ${PROJECT_NAME} pangolin ${Boost_LIBRARIES})
target_include_directories(app_dynamics_vio PRIVATE ${Boost_INCLUDE_DIRS} )
