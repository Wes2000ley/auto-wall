"""Build the optional C++ centerline tracer with CMake and a local C++ compiler."""

from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit("CMake was not found. Install CMake and a C++17 compiler first.")
    build = root / "build" / "native"
    subprocess.run([cmake, "-S", str(root / "native"), "-B", str(build),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run([cmake, "--build", str(build), "--config", "Release"], check=True)
    print("Built optional centerline backend in", root / "src" / "core" / "_native")


if __name__ == "__main__":
    main()
