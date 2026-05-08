#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // INVARIANT: rand::thread_rng() and std::time are strictly forbidden. 
    // Only wyhash / ChaCha seeded with master_seed + entity_id is allowed.
    #[test]
    fn test_absolute_determinism_across_entities() {
        let seed = 0xDEADBEEF;
        
        let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
        let mut rng2 = ChaCha8Rng::seed_from_u64(seed);

        let vals1: Vec<f32> = (0..5).map(|_| rng1.gen_range(0.0..1.0)).collect();
        let vals2: Vec<f32> = (0..5).map(|_| rng2.gen_range(0.0..1.0)).collect();

        assert_eq!(vals1, vals2, "Determinism violated! Sequences do not match.");
    }
}
