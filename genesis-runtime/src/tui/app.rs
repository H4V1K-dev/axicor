// genesis-runtime/src/tui/app.rs
use std::sync::Arc;
use std::time::Duration;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use super::DashboardState;

pub fn run_tui_thread(state: Arc<DashboardState>) {
    std::thread::spawn(move || {
        enable_raw_mode().unwrap();
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture).unwrap();
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).unwrap();

        loop {
            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Min(10),
                    ].as_ref())
                    .split(f.size());

                // Читаем атомики (Zero-cost snapshot)
                let ticks = state.total_ticks.load(std::sync::atomic::Ordering::Relaxed);
                let nights = state.night_count.load(std::sync::atomic::Ordering::Relaxed);
                let is_night = state.is_night_phase.load(std::sync::atomic::Ordering::Relaxed);
                let batch_ms = state.latest_batch_ms.load(std::sync::atomic::Ordering::Relaxed);
                let udp_in = state.udp_in_packets.load(std::sync::atomic::Ordering::Relaxed);

                let phase_str = if is_night { "🌙 NIGHT PHASE (Maintenance)" } else { "☀️ DAY PHASE (Hot Loop)" };
                let phase_color = if is_night { Color::Yellow } else { Color::Cyan };

                let header_text = format!(
                    " Genesis AGI Engine | Phase: {} | Ticks: {} | Nights: {} | UDP In: {}",
                    phase_str, ticks, nights, udp_in
                );

                let header = Paragraph::new(header_text)
                    .style(Style::default().fg(phase_color))
                    .block(Block::default().borders(Borders::ALL).title(" Global State "));
                f.render_widget(header, chunks[0]); // changed from `chunks` to `chunks[0]`

                let perf_text = format!(
                    "Last Batch Time: {} ms\nThroughput: {} ticks/sec\n\n[Press 'q' to gracefully shutdown]",
                    batch_ms,
                    if batch_ms > 0 { (100 * 1000) / batch_ms } else { 0 } // предполагая sync_batch_ticks = 1000? Let's just do an approx
                );

                let perf = Paragraph::new(perf_text)
                    .block(Block::default().borders(Borders::ALL).title(" Core Loop Performance "));
                f.render_widget(perf, chunks[1]);
            }).unwrap();

            // Опрос клавиатуры (non-blocking)
            if event::poll(Duration::from_millis(200)).unwrap() {
                if let Event::Key(key) = event::read().unwrap() {
                    if let KeyCode::Char('q') = key.code {
                        break;
                    }
                }
            }
        }

        // Cleanup
        disable_raw_mode().unwrap();
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        ).unwrap();
        terminal.show_cursor().unwrap();
        std::process::exit(0); // Жесткий выход при нажатии 'q'
    });
}
