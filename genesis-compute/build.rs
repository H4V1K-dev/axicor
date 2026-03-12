fn main() {
    println!("cargo:rerun-if-changed=src/cuda/");
    println!("cargo:rerun-if-changed=src/amd/");

    if cfg!(feature = "mock-gpu") {
        return; // Используем программные заглушки
    }

    if cfg!(feature = "amd") {
        cc::Build::new()
            .compiler("hipcc")
            .file("src/amd/bindings.hip")
            .file("src/amd/physics.hip")
            .flag("-O3")
            .flag("--offload-arch=gfx803") // <-- Флаг для архитектуры AMD Polaris (RX 470/480/570/580)
            .compile("genesis_amd");

        println!("cargo:rustc-link-search=native=/opt/rocm/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    } else {
        cc::Build::new()
            .cuda(true)
            .flag("-arch=sm_86") // NVIDIA Ampere и новее
            .flag("-O3")
            .flag("-allow-unsupported-compiler") // Разрешаем работу с GCC 14+ (хотя мы форсим 13)
            .flag("-ccbin=g++-13")             // Форсируем GCC 13 для совместимости с CUDA 12.4
            .file("src/cuda/bindings.cu")
            .file("src/cuda/physics.cu")
            .compile("genesis_cuda");

        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
