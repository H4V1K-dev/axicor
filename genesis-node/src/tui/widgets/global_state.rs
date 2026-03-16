use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Modifier},
    widgets::{Block, Borders, Paragraph, Gauge, BorderType},
    Frame,
};
use crate::tui::state::{DashboardState, Phase};

pub fn draw(f: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" GLOBAL SYSTEM STATE ")
        .border_type(BorderType::Plain);
    
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(16), // Logo
            Constraint::Min(40),    // Info & Progress
        ])
        .split(inner);

    // Left: Logo
    let logo_text = "  ┌─ GENESIS ─┐\n   AGI RUNTIME\n  DASHBOARD  ";
    let logo_p = Paragraph::new(logo_text)
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_widget(logo_p, chunks[0]);

    // Right: Info & Progress
    let info_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Uptime & Phase
            Constraint::Length(1), // Next Night
            Constraint::Length(1), // Progress bar
        ])
        .split(chunks[1]);

    let uptime_secs = state.uptime.elapsed().as_secs();
    let uptime_str = format!("UPTIME: {:02}h {:02}m {:02}s", uptime_secs / 3600, (uptime_secs % 3600) / 60, uptime_secs % 60);
    
    let phase_str = match state.global_phase {
        Phase::Day => "☀ DAY",
        Phase::Night => "🌙 NIGHT",
        Phase::Sleep => "💤 SLEEP",
    };
    let global_status = format!("{:<25} GLOBAL PHASE: {}   NIGHT #{}", uptime_str, phase_str, state.night_count);

    f.render_widget(Paragraph::new(global_status), info_chunks[0]);

    let ticks_in_phase = state.total_ticks % state.night_interval_ticks;
    let ticks_to_night = state.night_interval_ticks.saturating_sub(ticks_in_phase);
    let next_night_secs = if state.ticks_per_sec > 0.0 {
        (ticks_to_night as f64 / state.ticks_per_sec) as u64
    } else {
        0
    };
    
    let next_night_str = format!("Next night in: {:02}h {:02}m {:02}s", next_night_secs / 3600, (next_night_secs % 3600) / 60, next_night_secs % 60);
    f.render_widget(Paragraph::new(next_night_str), info_chunks[1]);

    let ratio = if state.night_interval_ticks > 0 {
        (ticks_in_phase as f64 / state.night_interval_ticks as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let gauge = Gauge::default()
        .block(Block::default())
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray))
        .ratio(ratio)
        .label(format!("DAY PHASE {:.1}%", ratio * 100.0));
    
    f.render_widget(gauge, info_chunks[2]);
}
