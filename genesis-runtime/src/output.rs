use genesis_core::constants::GXO_MAGIC;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct GxoMatrix {
    pub name_hash: u32,
    pub offset: u32,
    pub width: u16,
    pub height: u16,
    pub stride: u8,
}

#[derive(Debug)]
pub struct GxoFile {
    pub total_pixels: u32,
    pub matrices: Vec<GxoMatrix>,
    pub soma_ids: Vec<u32>, // Flat array: Pixel -> Soma Dense ID
}

impl GxoFile {
    pub fn load(path: &Path) -> Self {
        let bytes = std::fs::read(path).expect("Fatal: Failed to read .gxo file");
        assert!(bytes.len() >= 12, "Fatal: .gxo file too small");
                
        unsafe {
            let ptr = bytes.as_ptr();
            let magic = *(ptr as *const u32);
            assert_eq!(magic, GXO_MAGIC, "Fatal: Invalid .gxo magic bytes");

            let num_matrices = *(ptr.add(6) as *const u16) as usize;
            let total_pixels = *(ptr.add(8) as *const u32);

            let expected_size = 12 + (num_matrices * 16) + (total_pixels as usize * 4);
            assert_eq!(bytes.len(), expected_size, "Fatal: .gxo file size mismatch");

            let mut matrices = Vec::with_capacity(num_matrices);
            let descriptors_ptr = ptr.add(12);
            for i in 0..num_matrices {
                let desc_base = descriptors_ptr.add(i * 16);
                matrices.push(GxoMatrix {
                    name_hash: *(desc_base as *const u32),
                    offset: *(desc_base.add(4) as *const u32),
                    width: *(desc_base.add(8) as *const u16),
                    height: *(desc_base.add(10) as *const u16),
                    stride: *(desc_base.add(12) as *const u8),
                });
            }

            let payload_ptr = descriptors_ptr.add(num_matrices * 16) as *const u32;
            let mut soma_ids = Vec::with_capacity(total_pixels as usize);
            std::ptr::copy_nonoverlapping(payload_ptr, soma_ids.as_mut_ptr(), total_pixels as usize);
            soma_ids.set_len(total_pixels as usize);

            Self { total_pixels, matrices, soma_ids }
        }
    }
}
