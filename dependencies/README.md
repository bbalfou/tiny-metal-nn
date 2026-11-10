# Dependencies

This directory contains all native third-party dependencies. Existing project
source files and directories retain their layout. The root CMake build consumes these paths directly; both Python packages
use the same core CMake target.

| Dependency | Storage | Revision |
| --- | --- | --- |
| MetalRT | `metalrt/` submodule (private, SSH) | `2b665f799bb80c3f2254b6cf2c9428d755ac42ef` |
| fmt | `fmt/` submodule | 11.2.0 |
| pybind11_json | `pybind11_json/` submodule | `d0bf434be9d287d73a963ff28745542daf02c08f` |
| nlohmann JSON | `nlohmann/json.hpp`, upstream single header | v3.12.0 |
| PCG32 | `pcg32/pcg32.h`, upstream single header | `70099eadb86d3999c38cf69d2c55f8adc1f7fe34` |

`vendor-manifest.json` records exact commits for every dependency, plus upstream
paths and SHA-256 checksums for the copied headers. Submodule gitlinks are the
authoritative pins for submodules. No branch-following configuration is used.

Initialize submodules after cloning:

```sh
git submodule update --init --recursive
```

MetalRT requires GitHub SSH access to `bbalfou/metalrt`.

The build uses `dependencies/` as an include directory for
JSON and PCG32,
`dependencies/fmt/include/`,
`dependencies/pybind11_json/include/`, and `dependencies/metalrt/include/`.
Project sources and the binding adapter both include `nlohmann/json.hpp`
directly from the vendored upstream single header.
The adapter is included because the existing PyTorch binding imports it.
fmt requires its compiled implementation as well as headers.

To update a submodule, check out the intended commit inside its directory,
update the manifest, and stage its path in the parent repository. To update a
standalone header, copy it from an exact upstream commit and refresh its
manifest entry and checksum. Third-party files retain their upstream notices.

The Torch extension uses the pybind11 headers bundled with the selected PyTorch
installation to match its tensor-caster ABI. MLX uses nanobind from its Python
build environment.
The image example uses Pillow; the native library has no image I/O dependency.
