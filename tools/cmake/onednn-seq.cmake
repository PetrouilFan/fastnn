# Toolchain file used by the `cpu-onednn` feature's CMake build.
#
# `onednn-src` 0.1.24 runs oneDNN's CMake with no `ONEDNN_CPU_RUNTIME` define,
# and oneDNN's default is **OMP** when the compiler supports OpenMP. On this
# host that default produced a `libdnnl.a` whose generated `dnnl_config.h` said
# `DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP`, i.e. every primitive would spawn its own
# OpenMP team. The v3 runtime owns its threads (`SessionThreads`), so the
# engine parallelises by output-channel blocks and needs the primitive itself
# to be single-threaded: this file selects oneDNN's sequential CPU runtime.
#
# `cmake` 0.1 only forwards the cache variables it defines itself plus a small
# set of environment variables, so a toolchain file is the supported way to add
# one from outside `onednn-src`'s build script. Cargo sets `CMAKE_TOOLCHAIN_FILE`
# from `.cargo/config.toml`; an existing value in the environment wins, which is
# why `OneDnnEngine` still checks `dnnl_version()->cpu_runtime` and runs the
# blocks serially when a foreign build reports a threaded runtime.
#
# The file deliberately sets nothing else: CMake performs its own compiler
# detection after loading a toolchain file.

set(ONEDNN_CPU_RUNTIME "SEQ" CACHE STRING "oneDNN sequential CPU runtime" FORCE)
