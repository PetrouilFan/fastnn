//! `fastnn-infer`: the FastNN inference command-line tool.
//!
//! Build without the `python` feature so the extension-module linker mode does
//! not apply to a standalone binary:
//!
//! ```text
//! cargo build --release --no-default-features --features cli
//! cargo build --release --no-default-features --features cli,vulkan
//! ```

fn main() {
    fastnn::inference::cli::main_entry();
}
