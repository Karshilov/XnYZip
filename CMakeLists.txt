cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    XnYZip
    VERSION 0.1.0
    DESCRIPTION "Short description"
    HOMEPAGE_URL "https://example.com/"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----
find_package(fmt REQUIRED CONFIG)
find_package(Eigen3 REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)
find_package(TBB REQUIRED)

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/source/*.cpp"
)
list(FILTER SRC_FILES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/source/main.cpp$")
list(FILTER SRC_FILES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/source/mpi_driver.cpp$")

add_library(XnYZip_lib ${SRC_FILES})

target_include_directories(XnYZip_lib
    PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
           ${ZSTD_INCLUDE_DIRS}
           ${Eigen3_INCLUDE_DIRS}
)

target_compile_features(XnYZip_lib PUBLIC cxx_std_20)

target_link_libraries(XnYZip_lib
    PUBLIC
        fmt::fmt
        Eigen3::Eigen
        TBB::tbb
        OpenMP::OpenMP_CXX
        PkgConfig::ZSTD
)


# ---- Declare executable ----

add_executable(XnYZip_exe source/main.cpp)
add_executable(XnYZip::exe ALIAS XnYZip_exe)

set_property(TARGET XnYZip_exe PROPERTY OUTPUT_NAME XnYZip)

target_compile_features(XnYZip_exe PRIVATE cxx_std_20)

target_link_libraries(XnYZip_exe PRIVATE XnYZip_lib)

# ---- Optional MPI driver (distributed compression for scaling experiments) ----

option(BUILD_MPI "Build the MPI driver for distributed compression" OFF)
if(BUILD_MPI)
    set(MPI_CXX_COMPILER_NAME "mpicxx" CACHE STRING "mpicxx wrapper name")
    find_program(MPI_CXX_WRAPPER NAMES ${MPI_CXX_COMPILER_NAME})
    if(NOT MPI_CXX_WRAPPER)
        message(FATAL_ERROR "BUILD_MPI=ON but '${MPI_CXX_COMPILER_NAME}' not found in PATH")
    endif()
    execute_process(COMMAND ${MPI_CXX_WRAPPER} --showme:compile
        OUTPUT_VARIABLE MPI_CXX_COMPILE_FLAGS_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE MPI_COMPILE_RC)
    execute_process(COMMAND ${MPI_CXX_WRAPPER} --showme:link
        OUTPUT_VARIABLE MPI_CXX_LINK_FLAGS_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE MPI_LINK_RC)
    if(NOT MPI_COMPILE_RC EQUAL 0 OR NOT MPI_LINK_RC EQUAL 0)
        message(FATAL_ERROR "Failed to query MPI flags via ${MPI_CXX_WRAPPER} --showme")
    endif()
    separate_arguments(MPI_CXX_COMPILE_FLAGS UNIX_COMMAND "${MPI_CXX_COMPILE_FLAGS_RAW}")
    separate_arguments(MPI_CXX_LINK_FLAGS UNIX_COMMAND "${MPI_CXX_LINK_FLAGS_RAW}")
    message(STATUS "MPI compile flags: ${MPI_CXX_COMPILE_FLAGS}")
    message(STATUS "MPI link flags: ${MPI_CXX_LINK_FLAGS}")

    add_executable(XnYZip_mpi source/mpi_driver.cpp)
    set_property(TARGET XnYZip_mpi PROPERTY OUTPUT_NAME XnYZip_mpi)
    target_compile_features(XnYZip_mpi PRIVATE cxx_std_20)
    target_compile_options(XnYZip_mpi PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    # MPI link flags must follow the object files (as-needed strips otherwise),
    # so attach them via target_link_libraries instead of target_link_options.
    target_link_libraries(XnYZip_mpi PRIVATE XnYZip_lib ${MPI_CXX_LINK_FLAGS})
endif()

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT XnYZip_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of XnYZip"
  )
endif()

include(cmake/dev-mode.cmake)
