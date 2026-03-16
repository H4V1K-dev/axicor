use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Modifier},
    widgets::{Block, Borders, Paragraph, Gauge, BorderType},
    Frame,
};
use crate::tui::state::DashboardState;

pub fn draw(f: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" HARDWARE & I/O NETWORK ")
        .border_type(BorderType::Plain);
    
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // VRAM text + gauge + breakdown
            Constraint::Length(1), // empty
            Constraint::Length(1), // UDP IN
            Constraint::Length(1), // UDP OUT
            Constraint::Length(1), // empty
            Constraint::Length(1), // Alert
        ])
        .split(inner);

    let vram_ratio = if state.vram_total_mb > 0.0 {
        (state.vram_used_mb / state.vram_total_mb).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let vram_text = format!("VRAM: {:.0}MB / {:.0}MB", state.vram_used_mb, state.vram_total_mb);
    f.render_widget(Paragraph::new(vram_text), chunks[0]);

    let gauge_color = if vram_ratio > 0.8 { Color::Red } else { Color::Cyan };
    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(gauge_color).bg(Color::DarkGray))
        .ratio(vram_ratio)
        .label(format!("{:.1}%", vram_ratio * 100.0));
    
    // ratatui 0.29 requires height for gauge but chunk[0] has height 2, so the layout isn't perfect
    // well chunks[0] is for vram text, we can put gauge in a separate chunk or just let it fill 1 line
    // actually chunks[0] is length 2: line 0 = text, line 1 = gauge? No, Paragraph fills it.
    // Let's split again.
    let vram_split = Layout::default().direction(Direction::Vertical).constraints([
        Constraint::Length(1), // Title 
        Constraint::Length(1), // Gauge
        Constraint::Length(1), // Somas/Dendrites/Axons breakdown
    ]).split(chunks[0]);
    f.render_widget(Paragraph::new(format!("VRAM: {:.0}MB / {:.0}MB", state.vram_used_mb, state.vram_total_mb)), vram_split[0]);
    f.render_widget(gauge, vram_split[1]);
    
    let breakdown_text = format!("Soma: {:.0}MB | Dend: {:.0}MB | Axon: {:.0}MB", state.vram_somas_mb, state.vram_dendrites_mb, state.vram_axons_mb);
    f.render_widget(Paragraph::new(breakdown_text).style(Style::default().fg(Color::DarkGray)), vram_split[2]);

    f.render_widget(Paragraph::new(format!("UDP IN:  {} pkts", state.udp_in_packets)), chunks[2]);
    f.render_widget(Paragraph::new(format!("UDP OUT: {} pkts", state.udp_out_packets)), chunks[3]);

    if state.oversized_skips > 0 {
        let alert = Paragraph::new(format!("⚠ {} OVERSIZED", state.oversized_skips))
            .style(Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD));
        f.render_widget(alert, chunks[5]);
    }
}
