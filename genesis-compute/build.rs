// genesis-runtime/build.rs

/// Detect GPU compute capability via nvidia-smi and return sm_XX arch string.
/// Returns None if nvidia-smi fails or output is unparseable.
fn detect_gpu_arch() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let cap = String::from_utf8_lossy(&output.stdout);
    let cap = cap.trim().lines().next()?.trim();
    // Parse "8.9" -> sm_89, "7.5" -> sm_75
    let parts: Vec<&str> = cap.split('.').collect();
    if parts.len() != 2 {
        return None;
    }
    let major: u8 = parts[0].trim().parse().ok()?;
    let minor: u8 = parts[1].trim().parse().ok()?;
    Some(format!("sm_{}{}", major, minor))
}

fn main() {
    // When running with `--features mock-gpu` (CI, CPU-only tests),
    // skip the CUDA compilation entirely. mock_ffi.rs provides all symbols
    // via #[no_mangle] Rust functions, so no native lib is needed.
    if std::env::var("CARGO_FEATURE_MOCK_GPU").is_ok() {
        return;
    }

    println!("cargo:rerun-if-changed=src/cuda/");

    // GPU arch: CUDA_ARCH env > nvidia-smi auto-detect > sm_75 fallback
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| detect_gpu_arch().unwrap_or_else(|| {
        println!("cargo:warning=Could not detect GPU via nvidia-smi, using sm_75 (Turing). Set CUDA_ARCH to override.");
        "sm_75".to_string()
    }));
    println!("cargo:warning=Building CUDA for -arch={}", arch);

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-O3")
        .flag("-use_fast_math")
        .flag("-default-stream=per-thread")
        .flag(&format!("-arch={}", arch));

    // Host compiler: Linux uses g++-12, Windows uses MSVC (cl.exe) by default
    #[cfg(unix)]
    build.flag("-ccbin=g++-12");

    build
        .file("src/cuda/bindings.cu")
        .file("src/cuda/physics.cu")
        .compile("genesis_cuda");
}
