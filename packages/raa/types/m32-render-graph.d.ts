/**
 * One node of a render-graph document (S50). `type` is a catalog node-type id from
 * {@link RenderGraphCatalog.nodes}; `v` is the node-type version the parameters were written
 * for (absent = the catalog's current version). `params` holds only catalog parameter keys of
 * the type; an absent parameter takes the catalog default.
 */
export interface RenderGraphNode {
  /** Node id, `/^[A-Za-z][A-Za-z0-9_]{0,63}$/` (no `.`), unique in the document. */
  id: string;
  /** Catalog node-type id, for example `pass.forward`. */
  type: string;
  /** The node-type version the parameters were written for; absent = current. */
  v?: number;
  /** Values of the type's catalog parameters; absent keys take the catalog defaults. */
  params?: Record<string, number | boolean | string | number[]>;
  /** Input ports of a `compute.wgsl` node; other types empty. */
  inputs?: RenderGraphComputePort[];
  /** Output ports of a `compute.wgsl` node; other types empty. */
  outputs?: RenderGraphComputePort[];
  /** The editor's math graph of a `compute.wgsl` node; ignored by raa. */
  math?: unknown;
  /** The user WGSL of a `compute.wgsl` node. */
  wgsl?: string;
}

/** One input or output port of a `compute.wgsl` node (S59). */
export interface RenderGraphComputePort {
  key: string;
  type: 'image2d' | 'depth2d' | 'buffer';
  format?: string;
  size?: { scale: number } | { width: number; height: number };
}

/** One link of a render-graph document. `from` and `to` are `"<nodeId>.<portKey>"`. */
export interface RenderGraphLink {
  /** `"<nodeId>.<portKey>"` naming an output port. */
  from: string;
  /** `"<nodeId>.<portKey>"` naming an input port; an input takes at most one link. */
  to: string;
  /** The link reads the previous frame's image of its source. Default false. */
  history?: boolean;
}

/**
 * A render-graph document of format `raa.render-graph` version 1 (S50), accepted by
 * {@link Engine.setRenderGraph} as an object or as its UTF-8 JSON text, and returned by
 * {@link Engine.renderGraph}.
 */
export interface RenderGraphDoc {
  format: 'raa.render-graph';
  version: 1;
  /** 1..64 characters. */
  name: string;
  nodes: RenderGraphNode[];
  links: RenderGraphLink[];
  /** Editor layout; ignored by raa and returned verbatim by {@link Engine.renderGraph}. */
  editor?: unknown;
}

/**
 * One issue of a render-graph validation or apply stage (S52). `nodeId`, `port`, `param` and
 * `link` are present only when the code fills them; `link` is the document index of the link.
 */
export interface RenderGraphIssue {
  /** A `graph.<snake_case>` code, see the issue-code table of the documentation. */
  code: string;
  severity: 'error' | 'warning';
  /** The id of the node the issue is on, when the code names one. */
  nodeId?: string;
  /** The port key the issue is on, when the code names one. */
  port?: string;
  /** The parameter key the issue is on, when the code names one. */
  param?: string;
  /** The document index of the link the issue is on, when the code names one. */
  link?: number;
  /** What is wrong, with the actual values. */
  message: string;
  /** One concrete action that fixes it. */
  fix: string;
}

/** The result of {@link Engine.setRenderGraph}. */
export interface RenderGraphResult {
  /** True when the graph was accepted and is the active document. */
  ok: boolean;
  /** Every error and warning of validation and apply, in S52 order; empty when `ok`. */
  issues: RenderGraphIssue[];
  /** The number of features the accepted graph rebuilt. */
  rebuilt: number;
}

/**
 * The forward `lightingOutput: 'layers-interlaced'` state of the latest submitted frame
 * ({@link Engine.lightingInterlace}). `frame` is the real-frame counter n (advances only on
 * submitted frames); `real` is true on an R frame (n even: the forward evaluated the signals at
 * the pixels (x, y) with (x & 1, y & 1) = `phase`), false on a G frame (no signal evaluated,
 * the `skipped` signal passes did not record). `phase` is [[0,0],[1,1],[1,0],[0,1]][(n / 2) % 4],
 * or [(n / 2 + 2) % 4] on a G frame with `irradianceRate` 'phase60' (E sampled on G frames).
 * The rates are those set by `pass.lighting.reconstruct` (`shadowRate`, `irradianceRate`).
 */
export interface LightingInterlaceState {
  mode: boolean;
  frame: number;
  real: boolean;
  phase: [number, number];
  skipped: ('shadow.csm' | 'ao.gtao' | 'shadow.contact' | 'gi.ddgi')[];
  shadowRate: 'phase' | 'real' | 'every';
  irradianceRate: 'phase30' | 'phase60' | 'full';
}

/** The per-frame stats of {@link Engine.renderGraphStats}, from the latest rendered frame. */
export interface RenderGraphStats {
  /** The Engine render invocation of the latest frame; null before one. */
  frameIndex: number | null;
  nodes: {
    id: string;
    type: string;
    /** Whether the node ran in that frame. */
    ran: boolean;
    resources: {
      port: string;
      width: number;
      height: number;
      layers: number;
      mips: number;
      format: string;
      bytes: number;
    }[];
    /**
     * Nodes that keep a history and report it (`pass.lighting.reconstruct`): whether that
     * history was valid when the node last ran. Absent for every other node.
     */
    historyValid?: boolean;
  }[];
}

/** The eight catalog categories, in the rank order of {@link RenderGraphCatalog.categories}. */
export type RenderGraphCategory =
  | 'geometry'
  | 'lighting'
  | 'gi'
  | 'shadows'
  | 'post'
  | 'compute'
  | 'view'
  | 'output';

/** The seven port types, in the order of {@link RenderGraphCatalog.portTypes}. */
export type RenderGraphPortType =
  | 'image2d'
  | 'image2dArray'
  | 'image3d'
  | 'imageCube'
  | 'depth2d'
  | 'depth2dArray'
  | 'buffer';

/** A graph format as the catalog writes it: a WebGPU texture-format name, or `surface`. */
export type RenderGraphFormat =
  | 'r8unorm'
  | 'rg8unorm'
  | 'rgba8unorm'
  | 'rgba8unorm-srgb'
  | 'bgra8unorm'
  | 'r16float'
  | 'rg16float'
  | 'rgba16float'
  | 'r32float'
  | 'rg32float'
  | 'rgba32float'
  | 'r32uint'
  | 'rg32uint'
  | 'rgba32uint'
  | 'rgb10a2unorm'
  | 'rg11b10ufloat'
  | 'depth32float'
  | 'surface';

/** The kind of a catalog parameter, as it constrains its JSON value (S50). */
export type RenderGraphParamKind =
  | 'number'
  | 'int'
  | 'bool'
  | 'enum'
  | 'vec2'
  | 'vec3'
  | 'vec4'
  | 'string';

/** What a parameter change forces the engine to rebuild. */
export type RenderGraphRebuild = 'none' | 'pipeline' | 'resources' | 'engine';

/** A catalog parameter default: a scalar for the scalar kinds, an array for vec2..vec4. */
export type RenderGraphParamValue = number | boolean | string | number[];

/** The size of a catalog output port (S51): scaled to the surface, fixed, or from a param/input. */
export type RenderGraphPortSize =
  | { scale: number }
  | { width: number; height: number }
  | { param: string }
  | { input: string };

/** Versioned texture contract for an area shadow signal. */
export interface RenderGraphSignalContract {
  version: number;
  encoding: 'stratifiedPairs' | 'restirAggregate' | 'packedVisibility';
  usage: ('textureBinding' | 'storageBinding' | 'copySrc' | 'copyDst')[];
  samples: number;
  minLayers: number;
  maxLayers: number;
  cadence: 'current' | 'previous';
  activeRect: 'fullScene';
}

/** One input port of a catalog node type (S51). */
export interface RenderGraphCatalogInput {
  /** Port key, `/^[a-z][A-Za-z0-9]{0,31}$/`. */
  key: string;
  label: string;
  type: RenderGraphPortType;
  /** The accepted texture format, or null for any format of the port type. */
  format: RenderGraphFormat | null;
  /** Accepted signal contracts, when the input requires a specific encoding. */
  signals?: RenderGraphSignalContract[];
  /** A required input must be linked; absent must be false for an optional one. */
  required: boolean;
  /**
   * `"<nodeType>.<port>"` default sources in preference order: an unlinked input reads the
   * first one present in the graph.
   */
  defaultSource: string[];
  /** True when an unlinked input with no default source present binds the dummy. */
  absentOk: boolean;
  /** True when any link of the port type is accepted, not only a default source. */
  generic: boolean;
  /** True when the port accepts a history link. */
  history: boolean;
  /** The fixed fragment input slot index of a forward-class input, or null. */
  slot: number | null;
}

/** One output port of a catalog node type (S51). */
export interface RenderGraphCatalogOutput {
  /** Port key, `/^[a-z][A-Za-z0-9]{0,31}$/`. */
  key: string;
  label: string;
  type: RenderGraphPortType;
  /** The format the port produces, or null when the port type leaves it open. */
  format: RenderGraphFormat | null;
  /** Published signal contract, when the output has a defined encoding. */
  signals?: RenderGraphSignalContract[];
  size: RenderGraphPortSize;
  mips: number;
  layers: number;
}

/** One parameter of a catalog node type (S51); `default`, `min` and `max` are in its kind's units. */
export interface RenderGraphCatalogParam {
  /** Parameter key. */
  key: string;
  label: string;
  /** The inspector group label. */
  group: string;
  kind: RenderGraphParamKind;
  asset?: 'neural-model' | 'generated-frame-model';
  default: RenderGraphParamValue;
  /** The lower bound, or null when the code states none (`unstatedRange`). */
  min: number | null;
  /** The upper bound, or null when the code states none (`unstatedRange`). */
  max: number | null;
  /** The inspector step, or null. */
  step: number | null;
  /** The enum options, or null for the other kinds. */
  options: string[] | null;
  /** The unit label, empty when the value is unitless. */
  unit: string;
  rebuild: RenderGraphRebuild;
  /** True for the parameters the inspector shows first. */
  primary: boolean;
  /** True when the existing code states no range or meaning; `min`/`max` are then null. */
  unstatedRange: boolean;
  /** The `file:line` of the definition the default was read from. */
  source: string;
}

/** One node type of the catalog (S51). */
export interface RenderGraphCatalogNode {
  /** Node-type id: `pass.<pipeline entry>` for a pass, `compute.wgsl` for a compute node. */
  id: string;
  version: number;
  category: RenderGraphCategory;
  label: string;
  description: string;
  capability?: 'experimental-learned' | 'experimental-operator' | 'experimental-analytic';
  /** The pipeline entry name the node drives, or null for view, compute and output nodes. */
  pass: string | null;
  contributor: boolean;
  defaultOrder: number;
  multiInstance: boolean;
  sideEffect: boolean;
  /** A required node must be present in a graph (`pass.prepass`, `pass.forward`, `pass.tonemap`). */
  requiredInGraph: boolean;
  /** False for a pass node whose pipeline entry is not installed in this engine. */
  available: boolean;
  inputs: RenderGraphCatalogInput[];
  outputs: RenderGraphCatalogOutput[];
  params: RenderGraphCatalogParam[];
}

/** One starter graph or compute preset of the catalog, its document text in `json`. */
export interface RenderGraphCatalogEntry {
  name: string;
  file: string;
  json: string;
}

/**
 * The render-graph catalog (S51), returned by {@link Engine.renderGraphCatalog}: the one
 * source of node types, ports, parameters and defaults of the graph format. It is a fresh
 * object; changing it changes nothing in the engine.
 */
export interface RenderGraphCatalog {
  format: 'raa.render-graph-catalog';
  version: 1;
  /** The document format version the catalog describes: 1. */
  graphVersion: 1;
  categories: RenderGraphCategory[];
  portTypes: RenderGraphPortType[];
  nodes: RenderGraphCatalogNode[];
  starters: RenderGraphCatalogEntry[];
  presets: RenderGraphCatalogEntry[];
  frozen: string[];
}

/** Metadata of the graph and extent last accepted by the engine. */
export interface RenderGraphSelectionState {
  frozen: string | null;
  sceneSize: [width: number, height: number] | null;
  areaModel: string | null;
}

/**
 * The render-graph part of the Engine (S54). Every method throws `api.disposed` on a disposed
 * Engine. `setRenderGraph` never throws for an invalid graph: it returns `ok: false` with the
 * issues and the previous graph (or no graph) keeps rendering.
 */
export interface Engine {
  /** The catalog of the engine's installed passes, a fresh {@link RenderGraphCatalog}. */
  renderGraphCatalog(): RenderGraphCatalog;
  /**
   * Validates and installs a graph atomically. Accepts a {@link RenderGraphDoc} object or its
   * UTF-8 JSON text. `null` restores the no-graph configuration exactly. A refusal changes
   * nothing and names its nodes, ports, parameters or links in the returned issues.
   *
   * Throws `api.invalid_argument` for a value that is not an object, string or null, and
   * `api.disposed` after dispose.
   */
  setRenderGraph(graph: RenderGraphDoc | string | null): RenderGraphResult;
  /** Validates an FP32 area-pilot package and registers it under a unique per-engine name. */
  registerAreaRenderModel(name: string, manifest: string, weights: Uint8Array): void;
  /** Registers a verified FP32 ABI and weight hash. Model qualification remains experimental. */
  registerNeuralModel(id: string, manifest: string, weights: Uint8Array): void;
  /** Selects independent real-frame heads on an explicit graph, or restores its analytic path. */
  setNeuralReconstruction(options: NeuralModelSelection | null): void;
  neuralReconstructionStats(): NeuralModelStats | null;

  /**
   * Selects an editable graph, a frozen graph (pass null as graph), or the engine default.
   * Optional scene dimensions must be supplied together. A refused selection keeps the
   * previous graph, extent, and model. An area model requires a hardware adapter and a
   * full-canvas view with area.trace.raw linked directly to area.mask.src and that mask
   * feeding forward.areaShadow. Select the graph without a model to use raw visibility.
   */
  setRenderGraphSelection(
    graph: RenderGraphDoc | string | null,
    frozen?: string,
    sceneWidth?: number,
    sceneHeight?: number,
    areaModel?: string,
  ): RenderGraphResult;
  renderGraphSelection(): RenderGraphSelectionState;
  /**
   * The active document, canonical (S50), with its parameters as given. With no graph, the
   * export of today's configuration: one node per enabled pass and contributor in execution
   * order, every parameter written, the colour chain and an `out` output node.
   */
  renderGraph(): RenderGraphDoc;
  /** Whether a graph is set; false with no graph, including after `setRenderGraph(null)`. */
  renderGraphActive(): boolean;
  /** The issues of the last {@link Engine.setRenderGraph}, errors and warnings, in S52 order. */
  renderGraphIssues(): RenderGraphIssue[];
  /** The node stats of the latest rendered frame, a fresh {@link RenderGraphStats}. */
  renderGraphStats(): RenderGraphStats;
  /**
   * The setters the active graph owns, in this order: `setPassEnabled`, `setPassOrder`,
   * `setToneMapping`, `setShadows`, `setAo`, `setContactShadows`, `setBloom`,
   * `setTransparency`, `setSsr`, `setDof`, `setFog`, `setSunShafts`, `setGrading`,
   * `applyEnvironment`, `setCel`, `setVirtualGeometry`, `setAreaLightShading`,
   * `setAreaLightShadows`, `setGlobalIllumination`, `setGiQuality`.
   */
  renderGraphOwnedSetters(): string[];
  /**
   * The forward `layers-interlaced` state of the latest submitted frame, with or without a
   * graph; null before the first submitted frame. `mode` is false in every other
   * `lightingOutput`.
   */
  lightingInterlace(): LightingInterlaceState | null;
}

export interface NeuralModelHeads {
  area: boolean;
  reflection: boolean;
  sr: boolean;
}

export interface NeuralModelSelection extends Partial<NeuralModelHeads> {
  model: string;
}

export interface NeuralModelStats {
  model: string;
  kind: string;
  trainedHeads: ('area' | 'reflection' | 'display')[];
  weightsSha256: string;
  selected: NeuralModelHeads;
  ran: NeuralModelHeads;
  realFrame: number | null;
  sceneSize: [number, number];
  displaySize: [number, number];
  gpuBytes: number;
  dispatches: number;
  historyFrames: number;
}
