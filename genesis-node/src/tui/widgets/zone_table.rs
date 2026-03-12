use ratatui::{
    layout::{Constraint, Rect},
    style::{Color, Style, Modifier},
    widgets::{Block, Borders, Row, Cell, Table, BorderType},
    Frame,
};
use crate::tui::state::{DashboardState, Phase, FocusedPanel};

pub fn draw(f: &mut Frame, area: Rect, state: &mut DashboardState) {
    let border_color = if state.focus == FocusedPanel::ZoneTable { Color::Cyan } else { Color::DarkGray };
    let title = if state.focus == FocusedPanel::ZoneTable { "▶ PER-ZONE NEURAL TELEMETRY (Active) " } else { " PER-ZONE NEURAL TELEMETRY " };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(title)
        .border_type(BorderType::Plain);

    // Scrolling: show max 7 zones
    let visible_zones = state.zones.iter().skip(state.zone_scroll).take(7);

    let rows: Vec<Row> = visible_zones.map(|z| {
        // Phase + time-to-night
        let ticks_in_phase = state.total_ticks % z.night_interval_ticks.max(1);
        let ticks_to_night = z.night_interval_ticks.saturating_sub(ticks_in_phase);
        let secs_to_night = if state.ticks_per_sec > 0.0 {
            (ticks_to_night as f64 / state.ticks_per_sec) as u64
        } else {
            0
        };

        let phase_str = match z.phase {
            Phase::Day => {
                let m = secs_to_night / 60;
                let s = secs_to_night % 60;
                format!("DAY {:02}:{:02}", m, s)
            }
            Phase::Night => "NIGHT".to_string(),
            Phase::Sleep => "SLEEP".to_string(),
        };

        // Spike color
        let rate = z.spike_rate;
        let color = if rate < 1.0 {
            Color::Cyan
        } else if rate < 5.0 {
            Color::Yellow
        } else {
            Color::Red
        };

        // Visual activity bar
        let bars = (rate.min(10.0) / 1.0) as usize;
        let rate_str = format!("{:>5.2}% {}", rate, "█".repeat(bars));

        Row::new(vec![
            Cell::from(z.short_name.clone()),
            Cell::from(format!("{}", z.neuron_count)),
            Cell::from(format!("{}", z.axon_count)),
            Cell::from(rate_str).style(Style::default().fg(color)),
            Cell::from(phase_str),
        ])
    }).collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Min(15), 
            Constraint::Length(11), 
        ]
    )
    .header(
        Row::new(vec!["ZONE ID", "NEUR", "AXON", "ACTIVITY RATE", "PHASE"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1)
    )
    .block(block)
    .column_spacing(1);

    f.render_widget(table, area);
}
