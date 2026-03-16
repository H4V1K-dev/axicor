use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph, Sparkline, BorderType},
    Frame,
};
use crate::tui::state::DashboardState;

pub fn draw(f: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" CORE LOOP PERFORMANCE ")
        .border_type(BorderType::Plain);
    
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),    // Sparkline
            Constraint::Length(5), // Text stats below
        ])
        .split(inner);

    let data: Vec<u64> = state.wall_ms_history.iter().copied().collect();
    
    // Auto-scale: use max from actual data (+ 20% headroom), min 10
    let auto_max = data.iter().copied().max().unwrap_or(10).max(10);
    let sparkline_max = auto_max + (auto_max / 5).max(2); // +20% headroom

    let sparkline = Sparkline::default()
        .block(Block::default())
        .data(&data)
        .max(sparkline_max)
        .style(Style::default().fg(Color::Cyan));
    
    f.render_widget(sparkline, chunks[0]);

    let current_ms = data.last().copied().unwrap_or(0);
    let avg_ms = if data.is_empty() { 0 } else { data.iter().sum::<u64>() / data.len() as u64 };
    
    let tps = state.ticks_per_sec;
    let tps_str = if tps >= 1_000_000.0 {
        format!("{:.2}M t/s", tps / 1_000_000.0)
    } else if tps >= 1_000.0 {
        format!("{:.2}K t/s", tps / 1_000.0)
    } else {
        format!("{:.0} t/s", tps)
    };
    
    let text = format!(
        "Wall: {} ms/batch (avg: {})\nThroughput: {}\nBatch: #{}\nTicks: {}",
        current_ms,
        avg_ms,
        tps_str,
        state.batch_number,
        state.total_ticks,
    );
    
    f.render_widget(Paragraph::new(text), chunks[1]);
}
