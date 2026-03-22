use genesis_core::types::PackedPosition;

#[test]
fn test_packed_position_consistency() {
    let x = 123;
    let y = 456;
    let z = 78;
    let t = 9;
    let pos = PackedPosition::pack_raw(x, y, z, t);
    
    assert_eq!(pos.x() as u32, x);
    assert_eq!(pos.y() as u32, y);
    assert_eq!(pos.z() as u32, z);
    assert_eq!(pos.type_id(), t);
}
