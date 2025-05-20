cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    tonSZ
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

find_package(OpenMP REQUIRED)

add_library(tonSZ_lib INTERFACE)

target_include_directories(tonSZ_lib
    INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_compile_features(tonSZ_lib INTERFACE cxx_std_20)

target_link_libraries(tonSZ_lib
    INTERFACE fmt::fmt Eigen3::Eigen OpenMP::OpenMP_CXX ${ZSTD_LIBRARIES}
)

target_include_directories(tonSZ_lib INTERFACE ${ZSTD_INCLUDE_DIRS})


# ---- Declare executable ----

add_executable(tonSZ_exe source/main.cpp)
add_executable(tonSZ::exe ALIAS tonSZ_exe)

set_property(TARGET tonSZ_exe PROPERTY OUTPUT_NAME tonSZ)

target_compile_features(tonSZ_exe PRIVATE cxx_std_20)

target_link_libraries(tonSZ_exe PRIVATE tonSZ_lib)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT tonSZ_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of tonSZ"
  )
endif()

include(cmake/dev-mode.cmake)
