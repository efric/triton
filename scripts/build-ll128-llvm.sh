#!/usr/bin/env bash
set -euo pipefail

# Build the pinned AMD backend with LLVM #229501 until the dependency includes it.
TRITON_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LLVM_REVISION=6bc4aaf6641c56ee104f6200c96a6a571bc77052
LLVM_FIX=f8f362d165edce925b40468b84f2f68e7aa11f82
LLVM_PROJECT_PATH=${LLVM_PROJECT_PATH:-"$TRITON_ROOT/llvm-project-amd-ll128"}
LLVM_BUILD_PATH=${LLVM_BUILD_PATH:-"$LLVM_PROJECT_PATH/build"}
AMD_CODEGEN_BUILD_PATH=${AMD_CODEGEN_BUILD_PATH:-"$LLVM_PROJECT_PATH/codegen-build"}
LLVM_JOBS=${LLVM_JOBS:-32}
LLVM_C_COMPILER=${LLVM_C_COMPILER:-clang}
LLVM_CXX_COMPILER=${LLVM_CXX_COMPILER:-clang++}
LLVM_PATCH="$TRITON_ROOT/cmake/patches/llvm-cooperative-atomics.patch"
LLVM_PATCH_HASH=$(sha256sum "$LLVM_PATCH" | cut -d ' ' -f 1)

if [[ ! -e "$LLVM_PROJECT_PATH" ]]; then
  git init "$LLVM_PROJECT_PATH"
  git -C "$LLVM_PROJECT_PATH" fetch --depth=1 https://github.com/triton-lang/llvm-project.git "$LLVM_REVISION"
  git -C "$LLVM_PROJECT_PATH" checkout --detach FETCH_HEAD
fi
if [[ $(git -C "$LLVM_PROJECT_PATH" rev-parse HEAD) != "$LLVM_REVISION" ]]; then
  echo "Expected AMD LLVM $LLVM_REVISION at $LLVM_PROJECT_PATH" >&2
  exit 1
fi
if ! git -C "$LLVM_PROJECT_PATH" apply --reverse --check "$LLVM_PATCH" 2>/dev/null; then
  git -C "$LLVM_PROJECT_PATH" diff --exit-code
  git -C "$LLVM_PROJECT_PATH" diff --cached --exit-code
  git -C "$LLVM_PROJECT_PATH" apply --check "$LLVM_PATCH"
  git -C "$LLVM_PROJECT_PATH" apply "$LLVM_PATCH"
fi

cmake -G Ninja -S "$LLVM_PROJECT_PATH/llvm" -B "$LLVM_BUILD_PATH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$LLVM_C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$LLVM_CXX_COMPILER" \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_PROJECTS= \
  -DLLVM_TARGETS_TO_BUILD=AMDGPU \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_TESTS=ON \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_ENABLE_PIC=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_PARALLEL_LINK_JOBS=4
cmake --build "$LLVM_BUILD_PATH" --parallel "$LLVM_JOBS" --target \
  llvm-libraries llc opt llvm-as llvm-dis llvm-mc llvm-objdump llvm-readobj llvm-config \
  FileCheck count not split-file

cmake -G Ninja -S "$TRITON_ROOT/third_party/amd/backend/codegen" -B "$AMD_CODEGEN_BUILD_PATH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$LLVM_C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$LLVM_CXX_COMPILER" \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DLLVM_DIR="$LLVM_BUILD_PATH/lib/cmake/llvm" \
  -DTRITON_AMD_LLVM_REVISION="$LLVM_REVISION+ll128-$LLVM_FIX-${LLVM_PATCH_HASH:0:12}" \
  -DCMAKE_INSTALL_PREFIX="$TRITON_ROOT/third_party/amd/backend"
cmake --build "$AMD_CODEGEN_BUILD_PATH" --parallel "$LLVM_JOBS"
cmake --install "$AMD_CODEGEN_BUILD_PATH"

case $(uname -s) in
  Darwin) AMD_CODEGEN_LIBRARY=libtriton_amd_codegen.dylib ;;
  *) AMD_CODEGEN_LIBRARY=libtriton_amd_codegen.so ;;
esac
printf 'export TRITON_AMD_CODEGEN_PATH=%q\n' \
  "$TRITON_ROOT/third_party/amd/backend/lib/$AMD_CODEGEN_LIBRARY" > "$LLVM_PROJECT_PATH/env.sh"
echo "Source $LLVM_PROJECT_PATH/env.sh before configuring Triton or running kernels."
