/** Opaque wasm S19 graph retaining its rig and clips. Construct with loadGraph; free when done. */
export interface AnimationGraph {
  /** A fresh, independent state at the graph's entry states and declared defaults. */
  newState(): AnimationGraphState;
  /**
   * Advance `state` by exactly one tick. Refusals throw Error with code and path and leave
   * the state unchanged. A state of another graph is refused with graph.schema.
   */
  step(state: AnimationGraphState, params?: GraphParams | null): GraphStep;
  free(): void;
}

/** Opaque mutable execution state of one AnimationGraph. Independent of other states. */
export interface AnimationGraphState {
  /** A complete RGS2 snapshot, bound to its graph's rig and clip assets. */
  save(): Uint8Array;
  /** Replace this state from a snapshot. A refused snapshot keeps the live state. */
  restore(bytes: Uint8Array): void;
  free(): void;
}

/** A declared parameter value: a number for float and int, a boolean for bool and trigger. */
export type GraphParamValue = number | boolean;

/**
 * Per-tick inputs by declared parameter name. Omitted names keep their last applied value;
 * a trigger given as true pulses for this tick only, and false is ignored.
 */
export type GraphParams = Readonly<Record<string, GraphParamValue>>;

/** One clip event crossed this tick. `layer` and `clip` are zero-based indices. */
export interface GraphEvent {
  readonly name: string;
  readonly tick: number;
  readonly layer: number;
  readonly clip: number;
}

/** The output of one graph tick. */
export interface GraphStep {
  /** 16 column-major floats per rig bone in bone order: the S18 setSkinPose layout. */
  readonly skin: Float32Array;
  /** Extracted root translation delta (3 floats); rotation is always identity (4 floats). */
  readonly rootDelta: { readonly t: Float32Array; readonly r: Float32Array };
  readonly events: readonly GraphEvent[];
}

/** Additive surface to intersect with the brick's RaaAnimApi, next to RaaAnimClipsApi. */
export interface RaaAnimGraphApi {
  /**
   * Load strict S19 raa.animgraph v1 JSON against a rig and clip handles. The handles are
   * borrowed and remain usable. Graph references name each clip by its S17 `name`, which
   * must be unique across the supplied handles. Refusals are Error objects with code, path
   * and message; non-array or non-handle arguments throw TypeError with code
   * api.invalid_input.
   */
  loadGraph(json: string, rig: AnimationRig, clips: AnimationClip[]): AnimationGraph;
  /**
   * Run the shared 600-tick penguin script (`graph-run`). Seed is record metadata only and
   * ticks is capped at the script length; the record echoes both inputs as given.
   * Seed and ticks outside the u32 range throw TypeError with code api.invalid_input.
   */
  selftest(testCase: 'graph-run', seed: number, ticks: number): {
    case: 'graph-run'; seed: number; ticks: number; hash: string;
  };
}
