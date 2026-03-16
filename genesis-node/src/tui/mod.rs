pub mod state;
pub mod layout;
pub mod input;
pub mod widgets;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use anyhow::Result;
use crossterm::{
    event::{self, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    Terminal,
};
use state::DashboardState;

pub fn run_tui(telemetry: Arc<state::LockFreeTelemetry>, log_mode: bool) -> Result<()> {
    if log_mode {
        run_log_reporter(telemetry);
        return Ok(());
    }

    // [TUI] Redirect stdout/stderr to /dev/null to suppress println! from all threads (Unix only).
    // Boot-time prints (before this point) still go to terminal normally.
    // After TUI exits, we restore normally via LeaveAlternateScreen (crossterm handles this).
    #[cfg(unix)]
    unsafe {
        let devnull = libc::open(b"/dev/null\0".as_ptr() as *const libc::c_char, libc::O_WRONLY);
        if devnull >= 0 {
            libc::dup2(devnull, libc::STDOUT_FILENO);
            libc::dup2(devnull, libc::STDERR_FILENO);
            libc::close(devnull);
        }
    }

    // Initialize TUI — writes go to the alternate screen via crossterm's own fd
    enable_raw_mode()?;
    #[cfg(unix)]
    let tty = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")?;
    #[cfg(not(unix))]
    let tty = std::io::stdout();
    execute!(&tty, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(tty);
    let mut terminal = Terminal::new(backend)?;

    let res = run_app(&mut terminal, telemetry);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        // Can't eprintln here since stderr is /dev/null, but that's fine
        let _ = err;
    }
    
    Ok(())
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, telemetry: Arc<state::LockFreeTelemetry>) -> Result<()> {
    let tick_rate = Duration::from_millis(200);
    let mut local_state = DashboardState::new();

    // Open /dev/tty for keyboard events since stdin might be redirected (Unix only)
    #[cfg(unix)]
    {
        let _tty_fd = std::fs::File::open("/dev/tty")?;
        // crossterm reads from /dev/tty automatically on Linux when stdin is not a tty
    }

    loop {
        // 1. SYNC: Pull metrics from Lock-Free atomics
        local_state.batch_number = telemetry.batch_number.load(Ordering::Relaxed);
        local_state.total_ticks = telemetry.total_ticks.load(Ordering::Relaxed);
        local_state.udp_out_packets = telemetry.udp_out_packets.load(Ordering::Relaxed);
        
        let wall = telemetry.wall_ms.load(Ordering::Relaxed);
        local_state.push_wall_ms(wall);
        
        // Drain logs
        while let Some(log) = telemetry.logs.pop() {
            local_state.events.push_back(log);
            if local_state.events.len() > 200 {
                local_state.events.pop_front();
            }
        }

        // Update spikes
        for i in 0..16 {
            let hash = telemetry.zone_hashes[i].load(Ordering::Relaxed);
            if hash == 0 { break; }
            let spikes = telemetry.zone_spikes[i].load(Ordering::Relaxed);
            
            if let Some(z) = local_state.zones.iter_mut().find(|z| z.hash == hash) {
                z.spikes_last_batch = spikes;
            } else {
                // Discovery: Add new zone to local state if missing
                local_state.zones.push(state::ZoneMetrics {
                    hash,
                    name: format!("Zone_{:08X}", hash),
                    short_name: format!("{:08X}", hash),
                    neuron_count: 0,
                    axon_count: 0,
                    spikes_last_batch: spikes,
                    spike_rate: 0.0,
                    phase: state::Phase::Day,
                    night_interval_ticks: 0,
                });
            }
        }

        // 2. RENDER
        terminal.draw(|f| {
            layout::draw(f, &mut local_state);
        })?;

        if event::poll(tick_rate)? {
            if let Event::Key(key) = event::read()? {
                if input::handle_key(key, &mut local_state) {
                    break;
                }
            }
        }
    }
    
    Ok(())
}

fn run_log_reporter(telemetry: Arc<state::LockFreeTelemetry>) {
    let mut last_batch = 0;
    let mut local_state = DashboardState::new();
    
    loop {
        std::thread::sleep(Duration::from_millis(200));
        
        // SYNC
        let current_batch = telemetry.batch_number.load(Ordering::Relaxed);
        if current_batch == last_batch && local_state.is_running {
            continue; 
        }
        
        last_batch = current_batch;
        local_state.batch_number = current_batch;
        local_state.total_ticks = telemetry.total_ticks.load(Ordering::Relaxed);
        local_state.udp_out_packets = telemetry.udp_out_packets.load(Ordering::Relaxed);

        // Drain logs for sync
        while let Some(log) = telemetry.logs.pop() {
            local_state.events.push_back(log);
        }
        
        let wall_ms = telemetry.wall_ms.load(Ordering::Relaxed);
        let ticks = local_state.total_ticks;
        
        let now = chrono::Local::now().format("%H:%M:%S");
        eprintln!("[{}] [BATCH] #{} | {} ticks | {}ms wall", now, local_state.batch_number, ticks, wall_ms);
        
        // Simple log dump for zones
        for i in 0..16 {
            let hash = telemetry.zone_hashes[i].load(Ordering::Relaxed);
            if hash == 0 { break; }
            let spikes = telemetry.zone_spikes[i].load(Ordering::Relaxed);
            eprintln!("[{}] [ZONE]  {:08X}: {} spikes", now, hash, spikes);
        }
        
        eprintln!("[{}] [IO]    OUT: {}", now, local_state.udp_out_packets);
    }
}
