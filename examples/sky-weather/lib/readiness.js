export function createReadiness(stage) {
  let value = { status: 'pending', stage, error: null };
  return {
    snapshot: () => ({ ...value }),
    pending(nextStage) { value = { status: 'pending', stage: nextStage, error: null }; },
    ready() {
      if (value.status === 'failed') throw new Error('failed preparation requires a new pending preparation');
      value = { status: 'ready', stage: value.stage, error: null };
    },
    fail(failedStage, error) {
      value = { status: 'failed', stage: failedStage, error: error instanceof Error ? error.message : String(error) };
    },
  };
}
