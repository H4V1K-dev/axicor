fn main() {
    // 1.3.   Mock-GPU
    if std::env::var("CARGO_FEATURE_MOCK_GPU").is_ok() {
        return;
    }

    // [DOD FIX] Skip native build during 'cargo check' to allow C-ABI linting on non-GPU hosts.
    // CARGO_CFG_DEBUG is not reliable, but check vs build can be distinguished by specific env vars.
    // We use a pragmatic approach: if hipcc/nvcc are missing and we are not in a strict release build,
    // we emit a warning but don't fail.
    
    println!("cargo:rerun-if-changed=src/cuda/");
    println!("cargo:rerun-if-changed=src/amd/");

    if cfg!(feature = "amd") {
        if which::which("hipcc").is_err() {
            // [DOD FIX] Жесткий барьер. Никаких скрытых провалов.
            panic!("FATAL: hipcc compiler not found, but 'amd' feature is enabled. Check PATH or compile with 'mock-gpu'.");
        }

        cc::Build::new()
            .compiler("hipcc")
            .file("src/amd/bindings.hip")
            .file("src/amd/physics.hip")
            .flag("-O3")
            .flag("--offload-arch=gfx803")
            .compile("axicor_amd");

        if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-search=native=/opt/rocm/lib");
        }
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    } else if !cfg!(feature = "mock-gpu") {
        if which::which("nvcc").is_err() {
            // [DOD FIX] Если мы здесь, значит mock-gpu не включен. Требуем CUDA.
            panic!("FATAL: nvcc compiler not found, and 'mock-gpu' feature is NOT enabled. Either install CUDA Toolkit or build with '--features mock-gpu'.");
        }

        std::env::set_var("CXX", "g++-13");
        cc::Build::new()
            .cuda(true)
            .flag("-arch=sm_61")
            .flag("-O3")
            .flag("-w")
            .file("src/cuda/bindings.cu")
            .file("src/cuda/physics.cu")
            .compile("axicor_cuda");

        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
