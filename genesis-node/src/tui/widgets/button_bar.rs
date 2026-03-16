use ratatui::{
    layout::Rect,
    style::{Color, Style},
    widgets::Paragraph,
    Frame,
};
use crate::tui::state::DashboardState;

pub fn draw(f: &mut Frame, area: Rect, state: &DashboardState) {
    let start_stop = if state.is_running {
        "[⏹ Stop (F6)]"
    } else {
        "[▶ Start (F5)]"
    };

    let text = format!(
        " ──────{}──[+ Create (F7)]──[📂 Load (F8)]───────────────────────",
        start_stop
    );

    f.render_widget(
        Paragraph::new(text).style(Style::default().fg(Color::DarkGray)),
        area
    );
}
