Rust frontend of primitiv
=================================

Prerequisites
-------------

* Rust (1.22 or later)
* Clang (3.9 or later)
* (optional) CUDA (8.0 or later)

Install
---------------

```
mkdir work
cd work

# build primitiv-rust
git clone --branch develop https://github.com/primitiv/primitiv-rust/
cd primitiv-rust
cargo build [--features cuda]

# try xor example
cargo run --example xor
```

