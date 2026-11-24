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
        // MSVC's C++ runtime is linked by the compiler automatically, so only
        // add the libstdc++ dependency on GNU/Clang toolchains.
        let target = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
        if target != "msvc" {
            // `rustc-link-lib` reaches a dependent's link line; the extra
            // arguments reach this crate's own linkable targets. Both are
            // needed: on aarch64 GNU ld with Debian's default `--as-needed`
            // places the shared C++ runtime before the static oneDNN archive
            // and then drops it as unused, so `cargo test --lib` on that
            // target fails with undefined `operator new` and
            // `__class_type_info` unless the library is named again at the end
            // with `--no-as-needed` in force.
            println!("cargo:rustc-link-lib=dylib=stdc++");
            for kind in ["", "-tests"] {
                println!("cargo:rustc-link-arg{kind}=-Wl,--no-as-needed");
                println!("cargo:rustc-link-arg{kind}=-lstdc++");
            }
        }
    }
}
