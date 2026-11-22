//! Crate build script.
//!
//! Its only job is to make the optional oneDNN engine linkable. `onednn-src`
//! builds oneDNN as a **static C++ archive** (`libdnnl.a`) and emits
//! `cargo:rustc-link-lib=static=dnnl` itself, but Cargo does not add a C++
//! runtime for a Rust crate. When `cpu-onednn` is on we therefore ask the
//! linker for `libstdc++`; without the feature this script emits nothing and
//! no C++ library is referenced.
//!
//! `CARGO_FEATURE_CPU_ONEDNN` is set by Cargo for the enabled feature, which is
//! the documented way for a build script to observe it (`CARGO_CFG_*` carries
//! target cfgs, not features).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_CPU_ONEDNN").is_some() {
        // `libdnnl.a` is C++; its static initialisers and the oneDNN runtime
        // need the C++ standard library and the GCC runtime support routines.
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
