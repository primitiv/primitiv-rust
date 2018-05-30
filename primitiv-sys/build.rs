extern crate bindgen;

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};
use std::result::Result;

const FRAMEWORK_LIBRARY: &'static str = "primitiv";
const LIBRARY: &'static str = "primitiv_c";
const REPOSITORY: &'static str = "https://github.com/primitiv/primitiv";
const TAG: &'static str = "1ea9ba875ace79ca884ab448d7c0842d78d4b18d";

macro_rules! log {
    ($fmt:expr) => (println!(concat!("primitiv-sys/build.rs:{}: ", $fmt), line!()));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("primitiv-sys/build.rs:{}: ", $fmt), line!(), $($arg)*));
}
macro_rules! log_var(($var:ident) => (log!(concat!(stringify!($var), " = {:?}"), $var)));

fn main() {
    let lib_dir = env::var("PRIMITIV_LIBRARY_DIR").unwrap_or("/usr/local/lib".to_string());
    let library = format!("lib{}.so", LIBRARY);

    let find_library = Path::new(&lib_dir).join(&library).exists();
    let force_install = match env::var("PRIMITIV_FORCE_INSTALL") {
        Ok(s) => s != "0",
        Err(_) => false,
    };
    let force_build = match env::var("PRIMITIV_FORCE_BUILD") {
        Ok(s) => s != "0",
        Err(_) => false,
    };

    if !find_library || force_install || force_build {
        if force_build || true {
            // currently only support building from source.
            match build_from_src() {
                Ok(_) => log!("Successfully built `{}`.", library),
                Err(e) => panic!("Failed to build `{}`.\nreason: {}", library, e),
            }
        } else {
            match install_prebuild() {
                Ok(_) => log!("Successfully installed `{}`.", library),
                Err(e) => panic!("Failed to install `{}`.\nreason: {}", library, e),
            }
        }
    }

    exit(match build_bindings() {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}

fn install_prebuild() -> Result<((String, String)), Box<Error>> {
    Err("Not supported.".into())
}

fn build_from_src() -> Result<(), Box<Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?).join(format!("lib-{}", TAG));
    log_var!(out_dir);
    if !out_dir.exists() {
        fs::create_dir(out_dir.clone())?;
    }
    let framework_library_path = out_dir.join(format!("lib/lib{}.so", FRAMEWORK_LIBRARY));
    log_var!(framework_library_path);
    let library_path = out_dir.join(format!("lib/lib{}.so", LIBRARY));
    log_var!(library_path);
    if library_path.exists() && framework_library_path.exists() {
        log!(
            "{:?} and {:?} already exist, not building",
            library_path,
            framework_library_path
        );
    } else {
        let source =
            PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join(format!("target/source-{}", TAG));
        log_var!(source);
        if !source.exists() {
            run("git", |command| {
                command.arg("clone").arg(REPOSITORY).arg(&source)
            });
            run("git", |command| {
                command
                    .arg(format!(
                        "--git-dir={}",
                        &source.join(".git").to_str().unwrap()
                    ))
                    .arg("checkout")
                    .arg(TAG)
            });
        }
        let build_dir = source.join("build");
        if !build_dir.exists() {
            fs::create_dir(&build_dir)?;
        }
        let build_dir_s = build_dir.to_str().unwrap();
        run("cmake", |command| {
            command
                .arg(&source)
                .arg(format!("-B{}", build_dir_s))
                .arg(format!(
                    "-DCMAKE_INSTALL_PREFIX={}",
                    out_dir.to_str().unwrap()
                ))
                .arg("-DPRIMITIV_BUILD_C_API=ON")
                .arg(format!(
                    "-DPRIMITIV_USE_EIGEN={}",
                    if cfg!(feature = "eigen") { "ON" } else { "OFF" }
                ))
                .arg(format!(
                    "-DPRIMITIV_USE_CUDA={}",
                    if cfg!(feature = "cuda") { "ON" } else { "OFF" }
                ))
                .arg(format!(
                    "-DPRIMITIV_USE_OPENCL={}",
                    if cfg!(feature = "opencl") {
                        "ON"
                    } else {
                        "OFF"
                    }
                ))
        });
        run("make", |command| {
            command
                .arg(format!("--directory={}", build_dir_s))
                .arg("-j")
                .arg("4")
        });
        run("make", |command| {
            command.arg("install").arg(
                format!("--directory={}", build_dir_s),
            )
        });
    }
    env::set_var("PRIMITIV_LIBRARY_DIR", out_dir.join("lib"));
    env::set_var("PRIMITIV_INCLUDE_DIR", out_dir.join("include"));
    Ok(())
}

fn run<F>(name: &str, mut configure: F)
where
    F: FnMut(&mut Command) -> &mut Command,
{
    let mut command = Command::new(name);
    let configured = configure(&mut command);
    log!("Executing {:?}", configured);
    if !configured.status().unwrap().success() {
        panic!("failed to execute {:?}", configured);
    }
    log!("Command {:?} finished successfully", configured);
}

fn build_bindings() -> Result<(), Box<Error>> {
    let lib_dir = env::var("PRIMITIV_LIBRARY_DIR").unwrap_or("/usr/local/lib".to_string());
    let include_dir = env::var("PRIMITIV_INCLUDE_DIR").unwrap_or("/usr/local/include".to_string());
    println!("cargo:rustc-link-lib=dylib={}", LIBRARY);
    println!("cargo:rustc-link-search={}", lib_dir);

    let builder = bindgen::Builder::default()
        .clang_arg(format!("-I{}", include_dir))
        .header(format!("{}/primitiv/c/api.h", include_dir))
        .rustfmt_bindings(false)
        .generate_comments(false);

    builder
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(PathBuf::from(env::var("OUT_DIR")?).join("bindings.rs"))
        .expect("Couldn't write bindings!");
    Ok(())
}
