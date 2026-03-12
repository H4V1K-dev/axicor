use ratatui::{
    layout::{Constraint, Direction, Layout},
    Frame,
};
use crate::tui::state::DashboardState;
use crate::tui::widgets::{
    global_state, core_loop, zone_table, io_panel, button_bar, event_log
};

pub fn draw(f: &mut Frame, state: &mut DashboardState) {
    let area = f.area();

    // Main layout: Header(4), Metrics(10), Buttons(1), Expanding Log
    let vertical_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),   // Global System State
            Constraint::Length(10),  // Core / Zones / IO (Static height)
            Constraint::Length(1),   // Button Bar
            Constraint::Min(5),      // Event Log (Stretches down)
        ])
        .split(area);

    global_state::draw(f, vertical_chunks[0], state);
    
    // Middle section distribution
    let middle_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25), // Core Loop
            Constraint::Percentage(50), // Per-Zone
            Constraint::Percentage(25), // Hardware & I/O
        ])
        .split(vertical_chunks[1]);

    core_loop::draw(f, middle_chunks[0], state);
    zone_table::draw(f, middle_chunks[1], state);
    io_panel::draw(f, middle_chunks[2], state);

    button_bar::draw(f, vertical_chunks[2], state);
    event_log::draw(f, vertical_chunks[3], state);
}

