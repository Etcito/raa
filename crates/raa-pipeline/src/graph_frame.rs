//! Graph frames (S55): hooks a graph executor runs around the pipeline's frame loop.

use raa_core::RaaError;

use crate::{PrepareCtx, RecordCtx, ResourceTable};

/// Hooks called by the frame loop of a pipeline for each added graph frame, in insertion
/// order. Every method has a no-op default.
pub trait GraphFrame {
    /// Called once per frame after the resource table is cleared and before the first
    /// enabled pass's prepare.
    fn begin(&mut self, _table: &ResourceTable) {}

    /// Called before each enabled pass's `prepare`, with the pass entry name.
    fn before_prepare(&mut self, _pass: &str, _ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    /// Called before each enabled pass's `record`, with the pass entry name.
    fn before_record(&mut self, _pass: &str, _ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    /// Called after each enabled pass's `record`, with the pass entry name.
    fn after_record(&mut self, _pass: &str, _ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    /// Commits staged writes after the view's command buffer is submitted.
    fn submitted(&mut self) {}

    /// Invalidates retained history after a view discontinuity.
    fn invalidate_history(&mut self) {}

    /// Called once per frame after the last enabled pass's record.
    fn end(&mut self, _table: &ResourceTable) {}
}
