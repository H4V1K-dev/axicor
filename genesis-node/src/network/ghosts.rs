use std::path::Path;

/// Возвращает (src_indices, dst_indices)
pub fn load_ghosts(path: &Path) -> (Vec<u32>, Vec<u32>) {
    let bytes = std::fs::read(path).expect("Fatal: Failed to read .ghosts file");
    
    unsafe {
        let ptr = bytes.as_ptr();
        let count = *(ptr as *const u32) as usize;
        
        let src_ptr = ptr.add(4) as *const u32;
        let dst_ptr = ptr.add(4 + count * 4) as *const u32;
        
        let mut src = Vec::with_capacity(count);
        let mut dst = Vec::with_capacity(count);
        
        std::ptr::copy_nonoverlapping(src_ptr, src.as_mut_ptr(), count);
        std::ptr::copy_nonoverlapping(dst_ptr, dst.as_mut_ptr(), count);
        
        src.set_len(count);
        dst.set_len(count);
        
        (src, dst)
    }
}
