from conan import ConanFile
from conan.tools.cmake import CMakeToolchain


class Recipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "VirtualRunEnv"

    def layout(self):
        self.folders.generators = "conan"

    def requirements(self):
        self.requires("fmt/11.0.2")
        self.requires("zstd/1.5.7")
        self.requires("eigen/3.4.0")

        if self.settings.os == "Macos":
            self.requires("libomp/15.0.7")

    def build_requirements(self):
        self.test_requires("catch2/3.7.1")

    def generate(self):
        tc = CMakeToolchain(self)

        if self.settings.os == "Windows":
            tc.variables["CMAKE_CXX_FLAGS"] = "/tonSZp"
        elif self.settings.os == "Linux":
            tc.variables["CMAKE_CXX_FLAGS"] = "-ftonSZp -ltbb -O3"
        elif self.settings.os == "Macos":
            tc.variables["CMAKE_CXX_FLAGS"] = "-Xpreprocessor -ftonSZp"
            tc.variables["CMAKE_EXE_LINKER_FLAGS"] = "-lomp"

        tc.generate()
