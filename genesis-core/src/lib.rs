#![deny(warnings)]
#![deny(unused_variables)]
#![deny(dead_code)]
pub mod hash;
pub mod config;
pub mod constants;
pub mod coords;
pub mod ipc;
pub mod layout;
pub mod physics;
pub mod seed;
pub mod signal;
pub mod time;
pub mod types;

#[cfg(test)]
#[path = "test_gsop_math.rs"]
mod test_gsop_math;

#[cfg(test)]
mod test_tick;
