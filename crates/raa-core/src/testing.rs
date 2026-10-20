//! Native test helpers: a software-capable headless GPU and the SKIP convention.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::Thread;

use crate::{GpuContext, GpuOptions};

/// Headless GPU context that accepts a software adapter. On failure prints
/// `software_gpu: <reason>` to stderr and returns `None`.
pub fn software_gpu() -> Option<GpuContext> {
    let opts = GpuOptions {
        allow_software: true,
        ..GpuOptions::default()
    };
    match block_on(GpuContext::headless(&opts)) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("software_gpu: {e}");
            None
        }
    }
}

/// Prints `SKIP <test>: no software adapter (<reason>)` to stderr.
pub fn skip(test: &str, reason: &str) {
    eprintln!("SKIP {test}: no software adapter ({reason})");
}

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Runs a future to completion on the current thread (parks between polls).
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_on_returns_a_ready_value() {
        assert_eq!(block_on(async { 7 }), 7);
    }
}
