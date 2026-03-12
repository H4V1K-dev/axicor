use ratatui::{
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem, BorderType},
    Frame,
};
use crate::tui::state::{DashboardState, LogLevel, FocusedPanel};

pub fn draw(f: &mut Frame, area: Rect, state: &DashboardState) {
    let border_color = if state.focus == FocusedPanel::EventLog { Color::Cyan } else { Color::DarkGray };
    let title = if state.focus == FocusedPanel::EventLog { "▶ NIGHT PHASE EVENTS LOG (Active) " } else { " NIGHT PHASE EVENTS LOG " };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(title)
        .border_type(BorderType::Plain);

    // Dynamic height based on area
    let visible_lines = area.height.saturating_sub(2) as usize;

    let display_items: Vec<ListItem> = state.events.iter()
        .rev() // Start from latest
        .skip(state.log_scroll) // Apply scroll
        .take(visible_lines) // As much as fits
        .rev() // Reverse back for top-down display
        .map(|e| {
            let color = match e.level {
                LogLevel::Info => Color::Gray,
                LogLevel::Warning => Color::Yellow,
                LogLevel::Night => Color::LightMagenta,
                LogLevel::Error => Color::Red,
            };
            
            // Dopamine highlight
            let final_color = if e.message.contains("Dopamine") { Color::LightGreen } else { color };

            let content = format!("[{}] {}", e.timestamp, e.message);
            ListItem::new(content).style(Style::default().fg(final_color))
        })
        .collect();

    let list = List::new(display_items).block(block);
    f.render_widget(list, area);
}
