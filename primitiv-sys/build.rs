extern crate bindgen;

use std::env;
use std::error::Error;
use std::result::Result;
use std::process::exit;

fn main() {
    exit(match build() {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}

fn build() -> Result<(), Box<Error>> {
    let lib_dir = env::var("PRIMITIV_LIBRARY_DIR").unwrap_or("/usr/local/lib".to_string());
    let include_dir = env::var("PRIMITIV_INCLUDE_DIR").unwrap_or("/usr/local/include".to_string());
    println!("cargo:rustc-link-lib=dylib=primitiv_c");
    println!("cargo:rustc-link-search={}", lib_dir);

    let mut builder = bindgen::Builder::default()
        .clang_arg(format!("-I{}", include_dir))
        .header(format!("{}/primitiv/c/api.h", include_dir))
        .rustfmt_bindings(false)
        .generate_comments(false);

    if cfg!(feature = "cuda") {
        builder = builder.header(format!("{}/primitiv/c/api_cuda.h", include_dir));
    }

    builder
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");
    Ok(())
}
