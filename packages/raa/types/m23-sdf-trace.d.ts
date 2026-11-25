// M23 packed SDF tracing. build-package.sh merges this fragment into index.d.ts.
export interface TraceStats {
  sources: number;
  instances: number;
  bvhNodes: number;
  width: number;
  height: number;
  placeholders: number;
}

export interface Engine {
  /** Installs a complete little-endian TraceProgram v1 or v2 byte stream (v2 adds sampled SDF leaves).
   * Samples live in the packed data buffer; normals are evaluated from the SDF gradient.
   * A bent instance requires its matching packed bend record. A rejected program leaves the scene intact. */
  setTraceScene(program: Uint8Array): void;
  /** Replaces the 32-word records, preserving their bit patterns. The count and existing
   * bend-record associations must remain valid; a rejected update leaves the scene intact. */
  updateTraceInstances(instances: Float32Array): void;
  /** Removes the trace scene. */
  clearTraceScene(): void;
  /** Returns scene counts and full trace target dimensions. The world volume has a
   * separate longest-axis limit of 1024 voxels. */
  traceStats(): TraceStats;
}


/** Geometry reflections use the screen result first and trace only rejected pixels.
 * The full-frame path tracer is an explicit reference mode. */
export interface MeshTracingSettings {
  reflections?: boolean;
  reference?: boolean;
  confidenceThreshold?: number;
  reflectionBounces?: number;
  pathBounces?: number;
  /** Single-draw and independent 128-draw controls are intended for comparisons. */
  panelSampling?: 'stable' | 'single' | 'independent';
  diffuseOutput?: 'off' | 'direct' | 'direct-and-indirect';
  /** Selects the lighting contribution for the forward or full-frame mesh output. Default combined. */
  lightingOutput?: 'combined' | 'direct' | 'indirect' | 'bounce' | 'diffuse-indirect' | 'rough-specular' | 'rough-grid';
  /** With lightingOutput 'bounce' and reference enabled, selects an exact depth from 0 to 12. */
  lightingDepth?: number;
  pathMode?: 'off' | 'short' | 'traced' | 'learned';
  /** Live GI for secondary-hit shading in geometry reflections and learned tracing. Default auto. */
  indirectGrid?: 'auto' | 'off' | 'volume' | 'spherical';
  /** Experimental legacy sharp-mirror residency. 'measure' reports coverage without reserving slots. Default off. */
  mirrorPriority?: 'off' | 'measure' | 'on';
}

export interface MeshTracingStats extends Required<Pick<MeshTracingSettings,
  'confidenceThreshold' | 'reflectionBounces' | 'pathBounces'>> {
  frame: number;
  width: number;
  height: number;
  eligible: number;
  acceptedScreen: number;
  traced: number;
  screenMiss: number;
  offscreen: number;
  disocclusion: number;
  lowConfidence: number;
  nestedMirror: number;
  geometryHit: number;
  mirrorRays: number;
  pathRays: number;
  referencePixels: number;
  modelPixels: number;
  refreshPixels: number;
  diffuseMissingPixels: number;
  diffuseRays: number;
  specularRays: number;
  pathForegroundPixels: number;
  modelRejectedPixels: number;
  primaryBvhPixels: number;
  /** First real secondary rays and hits, before shading from a selected GI cache. */
  firstHitRays: number;
  firstHitHits: number;
  /** Order-independent digest of the dense triangle and world hit for a fixed scene and seed. */
  firstHitHash: number;
  sphericalDiffuseHits: number;
  volumeDiffuseHits: number;
  sphericalSpecularHits: number;
  volumeSpecularHits: number;
  sphericalDiffuseOffscreenInGrid: number;
  sphericalDiffuseOffscreenSupported: number;
  /** Current physical spherical support on non-mirror G-buffer pixels at trace resolution. */
  sphericalPrimaryInGrid: number;
  sphericalPrimarySupported: number;
  /** Offscreen hash-sampled receivers before the readback cap. */
  reflectedReceiverCandidates: number;
  directionalDiffuseHits: number;
  learnedDiffuseHits: number;
  gridDiffuseCaptureCandidates: number;
  diffuseMode: 'legacy' | 'analytic-directional' | 'learned';
  mirrorPriority: NonNullable<MeshTracingSettings['mirrorPriority']>;
  referenceSpp: number;
  hybridLobeSpp: number;
}

export interface Engine {
  /** Merges settings. Threshold 0–1, mirror bounces 1–6, reference path bounces 0–6. */
  setMeshTracing(options: MeshTracingSettings): void;
  /** Latest completed GPU counters for a mesh graph node, or for legacy tracing when omitted. */
  meshTracingStats(nodeId?: string): MeshTracingStats | null;
  /** Loads a specular model in validated 12→8→3 little-endian FP32 tensor order. */
  setMeshPathModel(weights: Uint8Array): void;
  /** Loads an immutable 8→8→1 FP32 gate in W1, b1, W2, b2 order. */
  setMeshGridDiffuseModel(weights: Uint8Array): void;
  /** Requests one in stride real secondary queries for the offline experiment. Zero disables. */
  setMeshGridDiffuseCapture(seed: number, stride: number): void;
  /** The latest completed readback. Each row has 128 words, at most 128 rows per frame. */
  meshGridDiffuseCapture(nodeId: string): {
    frame: number; seed: number; stride: number; width: number; height: number;
    candidates: number; rows: Uint32Array;
    /** One word per row. Four bits per interpolation corner. */
    cornerDiagnostics: Uint32Array;
  } | null;
  /** Reads the renderer's tight, little-endian linear RGBA16F scene color. */
  captureMeshLighting(nowMs: number, traced: boolean, raw?: boolean, evidenceSeed?: number): Promise<{
    size: [number, number];
    data: Uint8Array;
    /** The same-frame combined radiance while capturing an isolated forward view. */
    combinedData?: Uint8Array;
    /** Fixed-seed 48-byte records from this submitted fitted trace.
     * Each little-endian record contains world XYZ f32, first BVH triangle u32
     * (0xffffffff for a miss), original pre-grid specular RGB f32, grid trust f32,
     * accepted grid-tail RGB f32 and source u32 (0 none, 1 Volume, 2 Spherical). */
    roughEvidence?: { frame: number; size: [number, number]; data: Uint8Array };
  }>;
  /** Captures one fixed-seed specular RGB sample and guides from short or traced mode. */
  captureMeshPathSample(nowMs: number, seed: number): Promise<{ size: [number, number]; data: Uint8Array }>;
  /**
   * Offline readback of the latest submitted single view, without advancing its history.
   * Pass true to require the scene-linear output of the typed SSR replacement stage.
   */
  captureTrainingTaps(frameId: number, includeComposite?: boolean, includeArea?: boolean): Promise<{
    hdr: Uint8Array;
    depth: Uint8Array;
    normal: Uint8Array;
    motion: Uint8Array;
    reflectionHit?: Uint8Array;
    reflectionTrace?: Uint8Array;
    /** Zero validity and distance -2 indicate a typed trace leaving the screen. */
    reflectionMeta?: Uint8Array;
    reflectionIncident?: Uint8Array;
    reflectionReconstruction?: Uint8Array;
    compositeHdr?: Uint8Array;
    /** Scene-linear output on the resolve grid, when the graph publishes a TAA/SR output. */
    resolvedHdr?: Uint8Array;
    /** Packed weighted pairs for stochastic slots zero and one, RGBA16F. */
    areaPairs?: Uint8Array;
    areaReconstruction?: Uint8Array;
    areaMask?: Uint8Array;
    jitterNdc: [number, number];
  }>;
  /** Offline seeded reference trace, with an independent batch of 1 to 32 samples per pixel. */
  captureTrainingReference(nowMs: number, seed: number, samples: number): Promise<{
    size: [number, number];
    data: Uint8Array;
  }>;
}
