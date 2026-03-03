use super::*;

// use genesis_core::layout::VariantParameters;



#[test]
fn test_validate_physics_catches_panic() {
    // 0.6 m/s, 100 us, 25 um, 2 voxels -> v_seg = 1.2 (Panic!)
    let res = validate_physics_constraints(0.6, 100, 25.0, 2);
    assert!(res.is_err());
    assert!(res.unwrap_err().contains("CRITICAL INVARIANT BROKEN"));

    // 0.5 m/s, 100 us, 25 um, 2 voxels -> v_seg = 1.0 (Ok)
    let res = validate_physics_constraints(0.5, 100, 25.0, 2);
    assert!(res.is_ok());
    assert_eq!(res.unwrap(), 1);
}

#[test]
fn test_validate_blueprints_limit() {
    assert!(validate_blueprints(16).is_ok());
    assert!(validate_blueprints(17).is_err());
}

#[test]
fn test_distribute_quotas_perfect() {
    let quotas = vec![0.3, 0.7];
    let res = distribute_quotas(1000, &quotas).unwrap();
    assert_eq!(res, vec![300, 700]);
}

#[test]
fn test_distribute_quotas_rounding() {
    // 1000 * 0.33 = 330.0 -> floor=330
    // 1000 * 0.33 = 330.0 -> floor=330
    // 1000 * 0.34 = 340.0 -> floor=340
    let quotas = vec![0.33, 0.33, 0.34];
    let res = distribute_quotas(1000, &quotas).unwrap();
    assert_eq!(res, vec![330, 330, 340]);
    assert_eq!(res.iter().sum::<u32>(), 1000);

    // Кейс с реальным округлением
    // 100 * 0.333 = 33.3 -> 33
    // 100 * 0.333 = 33.3 -> 33
    // 100 * 0.334 = 33.4 -> 33
    // Total = 99. Remainder 1 goes to last active (idx 2).
    let quotas = vec![0.333, 0.333, 0.334];
    let res = distribute_quotas(100, &quotas).unwrap();
    assert_eq!(res, vec![33, 33, 34]);
    assert_eq!(res.iter().sum::<u32>(), 100);
}

#[test]
fn test_distribute_quotas_invalid_sum() {
    let quotas = vec![0.5, 0.4];
    let res = distribute_quotas(1000, &quotas);
    assert!(res.is_err());
}

#[test]
fn test_distribute_quotas_compensation_active_only() {
    // 100 * 0.5 = 50
    // 100 * 0.49 = 49
    // 100 * 0.01 = 1
    // Sum = 100. Let's force a remainder.
    // 10 * 0.33 = 3.3 -> 3
    // 10 * 0.33 = 3.3 -> 3
    // 10 * 0.0 = 0
    // 10 * 0.34 = 3.4 -> 3
    // Total = 9. Remainder 1 goes to idx 3 (last active).
    let quotas = vec![0.33, 0.33, 0.0, 0.34];
    let res = distribute_quotas(10, &quotas).unwrap();
    assert_eq!(res, vec![3, 3, 0, 4]);
    assert_eq!(res.iter().sum::<u32>(), 10);
}
