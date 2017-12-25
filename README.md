Rust frontend of primitiv
=================================

Prerequisites
-------------

* Rust (1.22 or later)
* Clang (3.9 or later)
* (optional) CUDA (7.5 or later)

Install
---------------

```
mkdir work
cd work

# build primitiv core with C API
git clone https://github.com/primitiv/primitiv/
cd primitiv
git checkout dc022e3fd4c343f7b46f2d04698f940211bca773
mkdir build
cd build
cmake .. -DPRIMITIV_BUILD_C_API=ON [-DPRIMITIV_USE_CUDA=ON]
make [-j <threads>]
[make install]

# build primitiv-rust
cd ../../
git clone --branch develop https://github.com/primitiv/primitiv-rust/
cd primitiv-rust
[PRIMITIV_INCLUDE_DIR=/path/to/work/primitiv PRIMITIV_LIBRARY_DIR=/path/to/work/primitiv/build/primitiv] cargo build [--features cuda]

# try xor example
[PRIMITIV_INCLUDE_DIR=/path/to/work/primitiv PRIMITIV_LIBRARY_DIR=/path/to/work/primitiv/build/primitiv] cargo build --example xor
[LD_LIBRARY_PATH=/path/to/work/primitiv/build/primitiv] cargo run --example xor
```

