use crossterm::event::{KeyCode, KeyEvent};
use crate::tui::state::{DashboardState, FocusedPanel};

pub fn handle_key(key: KeyEvent, state: &mut DashboardState) -> bool {
    match key.code {
        KeyCode::Char('q') => return true,
        KeyCode::Esc | KeyCode::F(6) => {
            state.is_running = false;
        }
        KeyCode::Char('s') | KeyCode::F(5) => {
            state.is_running = true;
        }
        KeyCode::Tab => {
            state.focus = if state.focus == FocusedPanel::ZoneTable {
                FocusedPanel::EventLog
            } else {
                FocusedPanel::ZoneTable
            };
        }
        KeyCode::Up => {
            match state.focus {
                FocusedPanel::EventLog => {
                    let max_scroll = state.events.len().saturating_sub(1);
                    if state.log_scroll < max_scroll {
                        state.log_scroll += 1;
                    }
                }
                FocusedPanel::ZoneTable => {
                    if state.zone_scroll > 0 {
                        state.zone_scroll -= 1;
                    }
                }
            }
        }
        KeyCode::Down => {
            match state.focus {
                FocusedPanel::EventLog => {
                    if state.log_scroll > 0 {
                        state.log_scroll -= 1;
                    }
                }
                FocusedPanel::ZoneTable => {
                    let max_scroll = state.zones.len().saturating_sub(1);
                    if state.zone_scroll < max_scroll {
                        state.zone_scroll += 1;
                    }
                }
            }
        }
        _ => {}
    }
    false
}
