use crate::tui::DashboardState;
use std::sync::Arc;
use std::time::Duration;

/// The Dashboard UI Application.
/// In this MVP, it serves as a background thread that prints the state
/// to the terminal if TUI is enabled, or simply holds the state.
pub struct DashboardApp {
    pub state: Arc<DashboardState>,
}

impl DashboardApp {
    pub fn new(state: Arc<DashboardState>) -> Self {
        Self { state }
    }

    /// Spawns the TUI rendering loop.
    /// [Stub] For now, it just sleeps and waits for events. 
    /// Rendering using Ratatui will be added in the final pass.
    pub fn spawn(self) {
        let state = self.state.clone();
        if !state.use_tui {
            return;
        }

        tokio::spawn(async move {
            println!("[Dashboard] TUI Initialized. Monitoring hot loop...");
            
            loop {
                // Future: Ratatui terminal.draw() here
                tokio::time::sleep(Duration::from_millis(200)).await;
                
                // For debug: occasionally print ticks to show it's alive
                let ticks = state.total_ticks.load(std::sync::atomic::Ordering::Relaxed);
                if ticks % 1000 == 0 && ticks > 0 {
                    // We don't want to spam stdout here, 
                    // but we can see the throughput atomics working.
                }
            }
        });
    }
}
