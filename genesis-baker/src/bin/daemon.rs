// genesis-baker/src/bin/daemon.rs
use genesis_baker::bake::axon_growth::GrownAxon;

fn main() {
    // This is a minimal stub for the baker-daemon which might be used for live interaction.
    println!("Genesis Baker Daemon Starting...");
}

/// Simulated daemon logic for incremental growth
pub fn incremental_grow(
    _neurons: &[(u32, u32, u32)], 
    _axons: &mut Vec<GrownAxon>,
    _new_counts: usize
) {
    // Placeholder for live sprouting logic if needed.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grown_axon_init() {
        let soma_idx = 0;
        let type_idx = 1;
        let tip_x = 100;
        let tip_y = 100;
        let tip_z = 100;
        let length_segments = 1;
        let segments = vec![0u32];
        
        // This is where the error was occurring
        let axon = GrownAxon { 
            soma_idx, 
            type_idx, 
            tip_x, 
            tip_y, 
            tip_z,
            length_segments, 
            segments,
            last_dir: glam::Vec3::ZERO,
        };
        
        assert_eq!(axon.tip_x, 100);
    }
}
