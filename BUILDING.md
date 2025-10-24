# Building with CMake

## Dependencies

For a list of dependencies, please refer to [conanfile.py](conanfile.py).

## Build

This project doesn't require any special command-line flags to build to keep
things simple.

Here are the steps for building in release mode with a single-configuration
generator, like the Unix Makefiles one:

```sh
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_TOOLCHAIN_FILE=conan/conan_toolchain.cmake
cmake --build build
```