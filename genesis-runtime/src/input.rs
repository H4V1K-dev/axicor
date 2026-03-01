use genesis_core::constants::GXI_MAGIC;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct GxiMatrix {
    pub name_hash: u32,
    pub offset: u32,
    pub width: u16,
    pub height: u16,
    pub stride: u8,
}

#[derive(Debug)]
pub struct GxiFile {
    pub total_pixels: u32,
    pub matrices: Vec<GxiMatrix>,
    pub axon_ids: Vec<u32>, // Flat array: Pixel → Virtual Axon ID
}

impl GxiFile {
    pub fn load(path: &Path) -> Self {
        let bytes = std::fs::read(path).expect("Fatal: Failed to read .gxi file");
        assert!(bytes.len() >= 12, "Fatal: .gxi file too small");

        unsafe {
            let ptr = bytes.as_ptr();

            let magic = u32::from_le_bytes(*(ptr as *const [u8; 4]));
            assert_eq!(magic, GXI_MAGIC, "Fatal: Invalid .gxi magic bytes");

            // Header layout (12 bytes):
            // [0..4]  magic     u32
            // [4]     version   u8
            // [5]     _padding  u8
            // [6..8]  num_matrices u16
            // [8..12] total_pixels u32
            let num_matrices = u16::from_le_bytes(*(ptr.add(6) as *const [u8; 2])) as usize;
            let total_pixels = u32::from_le_bytes(*(ptr.add(8) as *const [u8; 4]));

            let expected_size = 12 + (num_matrices * 16) + (total_pixels as usize * 4);
            assert_eq!(
                bytes.len(), expected_size,
                "Fatal: .gxi file size mismatch: got {} expected {}",
                bytes.len(), expected_size
            );

            // Matrix descriptors (16 bytes each):
            // [0..4]  name_hash u32
            // [4..8]  offset    u32
            // [8..10] width     u16
            // [10..12] height   u16
            // [12]    stride    u8
            // [13..16] _padding u8[3]
            let mut matrices = Vec::with_capacity(num_matrices);
            let descs_ptr = ptr.add(12);
            for i in 0..num_matrices {
                let d = descs_ptr.add(i * 16);
                matrices.push(GxiMatrix {
                    name_hash: u32::from_le_bytes(*(d as *const [u8; 4])),
                    offset:    u32::from_le_bytes(*(d.add(4) as *const [u8; 4])),
                    width:     u16::from_le_bytes(*(d.add(8) as *const [u8; 2])),
                    height:    u16::from_le_bytes(*(d.add(10) as *const [u8; 2])),
                    stride:    *d.add(12),
                });
            }

            // Payload: flat [total_pixels] u32 axon IDs
            let payload_ptr = descs_ptr.add(num_matrices * 16) as *const u32;
            let mut axon_ids = Vec::with_capacity(total_pixels as usize);
            std::ptr::copy_nonoverlapping(payload_ptr, axon_ids.as_mut_ptr(), total_pixels as usize);
            axon_ids.set_len(total_pixels as usize);

            Self { total_pixels, matrices, axon_ids }
        }
    }
}
