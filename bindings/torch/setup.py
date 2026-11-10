"""Build this Python package using the same CMake targets as the C++ API."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_helpers import copy_runtime_assets, environment_flag, native_root, project_version, snapshot_files

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = native_root(SCRIPT_DIR)
API = "torch"
PACKAGE = "tinymetalnn"


def switch(name):
    present = name in sys.argv
    if present:
        sys.argv.remove(name)
    return present


no_networks = switch("--no-networks") or environment_flag("TCNN_NO_NETWORKS", False)
no_jit = switch("--no-jit") or not environment_flag("TCNN_BUILD_WITH_RTC", True)


class CMakeBuild(build_ext):
    def build_extension(self, extension):
        if sys.platform != "darwin":
            raise RuntimeError("This binding requires macOS, Xcode, and a Metal-enabled framework")
        output_root = Path(self.get_ext_fullpath(extension.name)).resolve().parent.parent
        build_dir = Path(self.build_temp).resolve() / ("cmake-" + API)
        options = {
            "METALNN_BUILD_TORCH": API == "torch",
            "METALNN_BUILD_MLX": API == "mlx",
            "TCNN_NO_NETWORKS": no_networks,
            "TCNN_BUILD_WITH_RTC": not no_jit,
            "TCNN_HALF_PRECISION": environment_flag("TCNN_HALF_PRECISION", True),
            "TCNN_BUILD_USE_FAST_MATH": environment_flag("TCNN_BUILD_USE_FAST_MATH", False),
            "TCNN_BUILD_NO_FWD_BWD": environment_flag("TCNN_BUILD_NO_FWD_BWD", False),
        }
        arguments = ["cmake", "-S", str(ROOT_DIR), "-B", str(build_dir),
                     "-DCMAKE_BUILD_TYPE=" + ("Debug" if self.debug else "Release"),
                     "-DPython_EXECUTABLE=" + sys.executable,
                     "-DMETALNN_PYTHON_OUTPUT_DIRECTORY=" + str(output_root)]
        arguments += [f"-D{name}={'ON' if value else 'OFF'}" for name, value in options.items()]
        if os.environ.get("MACOSX_DEPLOYMENT_TARGET"):
            arguments.append("-DCMAKE_OSX_DEPLOYMENT_TARGET=" + os.environ["MACOSX_DEPLOYMENT_TARGET"])
        arguments += shlex.split(os.environ.get("CMAKE_ARGS", ""))
        subprocess.run(arguments, check=True)
        subprocess.run(["cmake", "--build", str(build_dir), "--target", "tinymetalnn_" + API,
                        "--config", "Debug" if self.debug else "Release", "--parallel",
                        str(self.parallel or os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4"))], check=True)
        copy_runtime_assets(ROOT_DIR, output_root / PACKAGE)


class BuildPythonPackage(build_py):
    def run(self):
        super().run()
        self._asset_outputs = copy_runtime_assets(ROOT_DIR, Path(self.build_lib) / PACKAGE)

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [str(p) for p in getattr(self, "_asset_outputs", [])]


class NativeSourceDistribution(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        release = Path(base_dir)
        staged = []
        for source in snapshot_files(ROOT_DIR):
            destination = release / "_native" / source.relative_to(ROOT_DIR)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            staged.append(destination.relative_to(release).as_posix())
        for manifest in release.glob("*.egg-info/SOURCES.txt"):
            entries = set(manifest.read_text().splitlines()) | set(staged)
            manifest.unlink()
            manifest.write_text("\n".join(sorted(entries)) + "\n")


setup(
    name=PACKAGE,
    version=project_version(ROOT_DIR),
    description="Native tiny-metal-nn PyTorch MPS bindings",
    long_description=(SCRIPT_DIR / "README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=["tinymetalnn", "tinymetalnn_bindings"],
    python_requires=">=3.10",
    install_requires=["torch>=2.7"],
    extras_require={"test": ["pytest>=7", "numpy"]},
    package_data={PACKAGE: ["rtc/include/**/*.metal"]},
    include_package_data=False,
    zip_safe=False,
    ext_modules=[Extension("tinymetalnn_bindings._C", sources=[])],
    cmdclass={"build_ext": CMakeBuild, "build_py": BuildPythonPackage, "sdist": NativeSourceDistribution},
)
