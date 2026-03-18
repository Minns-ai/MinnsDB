//! ScheduleRunner: cron-based trigger scheduling for WASM modules.

use serde::{Deserialize, Serialize};

use agent_db_core::types::Timestamp;

use crate::error::WasmError;
use crate::registry::ModuleRegistry;
use crate::runtime::WasmRuntime;

/// A scheduled task stored in ReDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleRecord {
    pub schedule_id: u64,
    pub module_name: String,
    pub cron: String,
    pub function: String,
    pub next_run: Timestamp,
    pub last_run: Option<Timestamp>,
    pub enabled: bool,
}

/// Manages scheduled tasks for WASM modules.
pub struct ScheduleRunner {
    schedules: Vec<ScheduleRecord>,
    next_schedule_id: u64,
}

impl Default for ScheduleRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl ScheduleRunner {
    pub fn new() -> Self {
        ScheduleRunner {
            schedules: Vec::new(),
            next_schedule_id: 1,
        }
    }

    /// Add a schedule.
    pub fn add(
        &mut self,
        module_name: String,
        cron_expr: String,
        function: String,
    ) -> Result<u64, WasmError> {
        let next_run = compute_next_run(&cron_expr)?;
        let id = self.next_schedule_id;
        self.next_schedule_id += 1;

        self.schedules.push(ScheduleRecord {
            schedule_id: id,
            module_name,
            cron: cron_expr,
            function,
            next_run,
            last_run: None,
            enabled: true,
        });

        Ok(id)
    }

    /// Remove a schedule.
    pub fn remove(&mut self, schedule_id: u64) -> bool {
        let before = self.schedules.len();
        self.schedules.retain(|s| s.schedule_id != schedule_id);
        self.schedules.len() < before
    }

    /// List schedules for a module.
    pub fn list_for_module(&self, module_name: &str) -> Vec<&ScheduleRecord> {
        self.schedules
            .iter()
            .filter(|s| s.module_name == module_name)
            .collect()
    }

    /// List all schedules.
    pub fn list_all(&self) -> &[ScheduleRecord] {
        &self.schedules
    }

    /// Check for due schedules and fire them. Returns number of schedules fired.
    pub fn tick(
        &mut self,
        runtime: &WasmRuntime,
        registry: &ModuleRegistry,
        now: Timestamp,
    ) -> usize {
        let mut fired = 0;

        for schedule in &mut self.schedules {
            if !schedule.enabled || schedule.next_run > now {
                continue;
            }

            // Fire the schedule
            if let Some(instance) = registry.get(&schedule.module_name) {
                let func_idx = instance
                    .descriptor
                    .functions
                    .iter()
                    .position(|f| f.name == schedule.function);

                if func_idx.is_some() {
                    // Safe truncation: schedule_id is monotonic from 1,
                    // and we cap at i32::MAX.
                    let sid = (schedule.schedule_id & 0x7FFFFFFF) as i32;
                    if let Err(e) = instance.call_schedule(runtime, sid) {
                        tracing::warn!(
                            "scheduled task {} on module '{}' failed: {}",
                            schedule.schedule_id,
                            schedule.module_name,
                            e
                        );
                    }
                } else {
                    tracing::warn!(
                        "schedule {} references unknown function '{}' on module '{}'",
                        schedule.schedule_id,
                        schedule.function,
                        schedule.module_name,
                    );
                }
            } else {
                tracing::debug!(
                    "schedule {} references unloaded module '{}'",
                    schedule.schedule_id,
                    schedule.module_name,
                );
            }

            schedule.last_run = Some(now);
            // Compute next run
            if let Ok(next) = compute_next_run(&schedule.cron) {
                schedule.next_run = next;
            } else {
                // Disable if cron is invalid
                schedule.enabled = false;
            }

            fired += 1;
        }

        fired
    }

    /// Restore schedules from persistence.
    pub fn restore(&mut self, schedules: Vec<ScheduleRecord>) {
        let max_id = schedules.iter().map(|s| s.schedule_id).max().unwrap_or(0);
        self.next_schedule_id = max_id + 1;
        self.schedules = schedules;
    }
}

/// Parse a cron expression and compute the next run timestamp (nanoseconds).
fn compute_next_run(cron_expr: &str) -> Result<Timestamp, WasmError> {
    let schedule = cron_expr
        .parse::<cron::Schedule>()
        .map_err(|e| WasmError::ScheduleError(format!("invalid cron '{}': {}", cron_expr, e)))?;

    let next = schedule
        .upcoming(chrono::Utc)
        .next()
        .ok_or_else(|| WasmError::ScheduleError("no upcoming execution".into()))?;

    Ok(next.timestamp_nanos_opt().unwrap_or(0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_schedule() {
        let mut runner = ScheduleRunner::new();
        // Every minute
        let id = runner
            .add("test-module".into(), "0 * * * * *".into(), "tick".into())
            .unwrap();
        assert_eq!(id, 1);
        assert_eq!(runner.list_all().len(), 1);
    }

    #[test]
    fn test_remove_schedule() {
        let mut runner = ScheduleRunner::new();
        let id = runner
            .add("test-module".into(), "0 * * * * *".into(), "tick".into())
            .unwrap();
        assert!(runner.remove(id));
        assert!(runner.list_all().is_empty());
    }

    #[test]
    fn test_invalid_cron() {
        let mut runner = ScheduleRunner::new();
        let result = runner.add("test".into(), "not a cron".into(), "tick".into());
        assert!(result.is_err());
    }
}
