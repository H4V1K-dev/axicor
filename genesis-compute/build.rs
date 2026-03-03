// genesis-runtime/build.rs
fn main() {
    // When running with `--features mock-gpu` (CI, CPU-only tests),
    // skip the CUDA compilation entirely. mock_ffi.rs provides all symbols
    // via #[no_mangle] Rust functions, so no native lib is needed.
    if std::env::var("CARGO_FEATURE_MOCK_GPU").is_ok() {
        return;
    }

    println!("cargo:rerun-if-changed=src/cuda/");

    cc::Build::new()
        .cuda(true)
        .flag("-O3")
        .flag("-use_fast_math")
        // TODO for 1080ti: Если у тебя архитектура отличная от Ampere (RTX 30xx/A100),
        // поменяй sm_80 на свою (например, sm_75 для Turing, sm_89 для Ada)
        .flag("-arch=sm_61")
        // Жёстко привязываем хост-компилятор, чтобы избежать конфликтов с GCC 14
        .flag("-ccbin=g++-12")
        .file("src/cuda/bindings.cu")
        .compile("genesis_cuda");
}
