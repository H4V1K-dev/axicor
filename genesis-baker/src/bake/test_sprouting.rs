use genesis_core::types::PackedPosition;

#[test]
fn test_sprouting_position_unpacking() {
    let x = 512;
    let y = 256;
    let z = 100;
    let t = 2;
    let packed = PackedPosition::pack_raw(x, y, z, t).0;
    
    let pos = PackedPosition(packed);
    assert_eq!(pos.x() as u32, x);
    assert_eq!(pos.y() as u32, y);
    assert_eq!(pos.z() as u32, z);
    assert_eq!(pos.type_id(), t);
}
