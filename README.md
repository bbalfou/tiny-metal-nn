# tiny-metal-nn

tiny-metal-nn is a port of tiny-cuda-nn to Apple Metal for neural networks and
input encodings on Apple Silicon.

The fused GPU kernels and encodings have been optimised. The GEMM and JIT fusion
implementations still require optimisation.

## Build

Requires macOS on Apple Silicon, Xcode with an SDK supporting MSL 4.1, CMake 3.27
or newer, and a C++17 compiler. 


Due to lack of access to other device the gpu kernels have been optimised for M3 Max 40-core chip. Milage on other chips may vary they and require architecture  to be greater than apple9


From the repository root:

```sh
git submodule update --init --recursive
cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --parallel
```
