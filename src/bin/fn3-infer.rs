//! `fn3-infer`: the FastNN 3.0 inference command-line tool (FN3-17 / #63).
//!
//! Build without the default features so the Python extension module does not
//! impose its linker mode on a standalone binary:
//!
//! ```text
//! cargo build --release --no-default-features --features cli
//! cargo build --release --no-default-features --features cli,vulkan
//! ```
//!
//! Run `fn3-infer help`, or read `docs/quickstart-fn3.md`.

fn main() {
    fastnn::fn3::cli::main_entry();
}
