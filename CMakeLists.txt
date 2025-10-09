cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    XnYSZ
    VERSION 0.1.0
    DESCRIPTION "Short description"
    HOMEPAGE_URL "https://example.com/"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)
find_package(TBB REQUIRED)

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/source/*.cpp"
)

add_library(XnYSZ_lib ${SRC_FILES})

target_include_directories(XnYSZ_lib
    PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
           ${ZSTD_INCLUDE_DIRS}
)

target_compile_features(XnYSZ_lib PUBLIC cxx_std_20)

target_link_libraries(XnYSZ_lib
    PUBLIC
        fmt::fmt
        Eigen3::Eigen
        TBB::tbb
        OpenMP::OpenMP_CXX
        ${ZSTD_LIBRARIES}
)


# ---- Declare executable ----

add_executable(XnYSZ_exe source/main.cpp)
add_executable(XnYSZ::exe ALIAS XnYSZ_exe)

set_property(TARGET XnYSZ_exe PROPERTY OUTPUT_NAME XnYSZ)

target_compile_features(XnYSZ_exe PRIVATE cxx_std_20)

target_link_libraries(XnYSZ_exe PRIVATE XnYSZ_lib)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT XnYSZ_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of XnYSZ"
  )
endif()

include(cmake/dev-mode.cmake)
