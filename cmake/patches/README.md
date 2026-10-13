# LLVM prerequisite for cooperative atomics

`llvm-cooperative-atomics.patch` backports LLVM commit
`f8f362d165edce925b40468b84f2f68e7aa11f82`
([LLVM #229501](https://github.com/llvm/llvm-project/pull/229501))
to Triton's AMD LLVM pin,
`6bc4aaf6641c56ee104f6200c96a6a571bc77052`.
The patch SHA-256 is
`dc335954f7fc9e470f35c5271666f2595bab0d91686a49c489edf456ec9592ba`.

The backport also supplies the prerequisite SelectionDAG patterns for that
pin. It adapts the v2i32/v4i32 global and flat parts of upstream commit
`3eb867f9b63ae2881bbe3c25d3c12842f2bdc55e`
([LLVM #229688](https://github.com/llvm/llvm-project/pull/229688)) without
backporting its wider atomic pattern refactor. Upstream regression cases are
retained, with golden checks regenerated for this pin's instruction scheduling
and register allocation.

The old backend assigns integer memory types to the vector-valued 16x8B and
8x16B intrinsics. Extracting flag components or retaining the loaded payload
across polling control flow can then crash value tracking. The patch uses
the intrinsic value type as its memory type and includes upstream tests for
SelectionDAG and GlobalISel.

Build an isolated, assertions-enabled AMD LLVM and the standalone codegen
library from the Triton checkout:

```sh
LLVM_JOBS=32 scripts/build-ll128-llvm.sh
source llvm-project-amd-ll128/env.sh
make dev-install-triton
make
```

The script supports `LLVM_PROJECT_PATH`, `LLVM_BUILD_PATH`,
`AMD_CODEGEN_BUILD_PATH`, `LLVM_C_COMPILER`, and `LLVM_CXX_COMPILER` overrides.
It creates a new source checkout when needed, checks the base revision, and
applies the patch without resetting existing source. The build follows the
static-library, PIC and assertion settings of the AMD LLVM CI package.
It also builds `llc`, `opt`, `llvm-as`, `llvm-dis`, `llvm-mc`, `llvm-objdump`,
`llvm-readobj`, `llvm-config`, `FileCheck`, `count`, `not`, and `split-file`
for compiler validation.

The installed library is local to this Triton checkout at
`third_party/amd/backend/lib/libtriton_amd_codegen.so` (or `.dylib`). Keep
`TRITON_AMD_CODEGEN_PATH` from `env.sh` set during Triton configuration and
kernel compilation so the dependency helper retains the patched library.
The library reports the AMD base commit, upstream fix commit, and local patch
hash through `get_amd_codegen_revision()`:

```sh
PYTHONPATH=python python -c 'from triton.backends.amd.compiler import get_amd_codegen_path, get_amd_codegen_revision; print(get_amd_codegen_path()); print(get_amd_codegen_revision())'
```

Triton's core LLVM pin remains unchanged. Core LLVM translates MLIR, runs the
first IR optimization pipeline, and serializes bitcode. The standalone AMD
library performs subsequent optimization, instruction selection and assembly
for normal AMD compilation.
The optional MIR dump/swap path invokes core LLVM code generation and still
requires the same fix in a separately built core LLVM if used for these
intrinsics. Retire this patch and local build recipe when the published AMD
LLVM dependency contains the upstream fix.
