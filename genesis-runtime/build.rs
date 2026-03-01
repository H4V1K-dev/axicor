// genesis-runtime/build.rs
fn main() {
    println!("cargo:rerun-if-changed=cuda/");

    cc::Build::new()
        .cuda(true)
        .flag("-O3")
        // Разрешаем агрессивные оптимизации регистров
        .flag("-use_fast_math") 
        // TODO for 1080ti: Если у тебя архитектура отличная от Ampere (RTX 30xx/A100), поменяй sm_80 на свою (например, sm_75 для Turing, sm_89 для Ada)
        .flag("-arch=sm_61") 
        // Жёстко привязываем хост-компилятор, чтобы избежать конфликтов с GCC 14
        .flag("-ccbin=g++-12") 
        .file("cuda/bindings.cu")
        .compile("genesis_cuda");
}
