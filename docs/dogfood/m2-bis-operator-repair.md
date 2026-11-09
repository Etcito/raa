M2 bis operator repair for card 7335

Evidence and refs

Execution 13283 verified RAA commit 2d1a585ca847 in 7335-opverify.
Its complete checks_json records build (48.11 s), functional (29.66 s),
select (14 pairs) and clean passing. Render failed after 337.39 s with
“GPU job running timeout after 300s”; its capture inventory is empty.
Fresh, judge, command-check, cargo and milestone-check have no results.
The timeout alone does not establish the underlying disconnect cause.

Fix 5341 attempt 1 introduced creating-adapter metadata at 2d1a585ca847;
its GPU verification then reached a different failure. Attempt 2 identified
an offscreen iframe and the preceding execution 13282 reports a 295 s
page timeout at measure:off. Neither attempt proved the full chain.
The current nomorechat runtime frameStyle uses left:0;top:0. The separate
interrupted-job replay issue reported in the previous operator attempt
was not repaired by this project change and has no passing verification.

Execution 13284 instead verified 7bd585059ced, containing command cleanup
only. Build, functional, select and clean passed; render failed in 6.28 s
at area/on with “hardware adapter required; software/fallback adapter
refused”. All five downstream steps again have no recorded results.
That verification did not include 2d1a585ca847's adapter repair.
Its archived render evidence has no captures. The surviving card capture
from 13281 has four blank identity strings, hardware:true, fallback:false,
and gpuErrors:[]. It is historical evidence, not a new GPU proof.
Current read-only GET /api/gpu/status reports a Windows Chrome hardware
client with vendor:nvidia, architecture:ampere, empty description and
software:false. This is client registration, not a capture of our commit.

Failure chain and changes

Adapter rejection <- missing identity in the returned page stats <-
wgpu 30 map_adapter_info copies only WebGPU description into name and
sets vendor/device to zero <- RAA AdapterSummary receives no vendor or
architecture and cannot reconstruct identity when description is blank.
This is a project metadata defect. Restore the prior unproven adapter
repair explicitly: preserve the raw info of the SAME adapter that creates
each device, separately for each measured variant. Keep blank identity,
software, fallback, browser, timing, image and judgment checks intact.
The blank-identity diagnostic now distinguishes this from software.

The requested whole-flow repair changes render from one job per pair to
one job for the complete selection. The page catches each step's failure,
collects page errors, rejected promises, console errors, failed resource
and fetch requests, and returns completed captures plus every step result.
The transport validates every pair, reports all failures, and leaves the
run incomplete if any step or diagnostic fails. Freshness binds each pair
to that original whole-flow result and the same host job/client identity.
No timeouts, milestone thresholds or downstream dependency edges changed.
No MAMA, SDFM, nomorechat source, held cards or milestone state changed.

Verification boundary

The functional command in .nmc/commands.json includes the transport, page
and many-lights Node tests. Controlled producers test the protocol without
rendering or launching any browser. These tests do not constitute GPU proof.
The supervisor must verify the committed repair through the managed runner:
build, functional, select, clean, ONE render job, fresh, judge, command-check,
cargo-check and milestone-check, with an actual passing result for every
selected entry. No card rerun is authorized by this report before that passes.
The operator shell has no GPU client/expose route and did not start a GPU
job or any browser. Render and the downstream checks remain unverified.

Operator 5363: controlling functional acceptance (owner decisions 5353–5358)

Measured chain: execution 13286 passed build, functional tests, selection of 14
pairs, clean, render and freshness. Judgment rejected area/on at 4/10 (samples
4,4,4; required 6), citing background, warmth and inability to infer emitter
sizes, equal intensity or prepass behavior from the PNG. It recorded no further
judgments and skipped command-check, cargo-check and milestone-check. Operator
5349 committed 6f3e221 to change sky presentation; verification 13287 then
rejected attached-light/on at 5/10 (5,5,6), citing marker colour and inability
to infer lifecycle/prepass behavior from one PNG. That commit was cancelled
and is not an ancestor of this checkout. Do not stack its aesthetic changes.
The current base ea5e460 preserves the adapter and one-job repairs, proven
through render/fresh in 13286; its whole command chain has not passed.

Failure <- unsuitable description-based image criteria <- technical scene notes
and multi-frame claims passed as single-image acceptance, with default look
judging <- missing separation of functional visual acceptance from deterministic
requirements. Fix the producer and judge call: scene metadata now supplies a
functional evidence label and descriptive filename from lights-evidence.js.
Explanatory scene descriptions still document no local shadows and the
transparent/displaced-fur prepass approximation. Those notes, numerical sizes,
intensity equality, API behavior and frame counts are not image criteria.
The full scene/API/transport Node suite remains mandatory and now runs in the
functional entry, including the existing area equality/extents tests. FPS,
canvas, hardware, browser, warmup, frames and provenance checks remain intact.
No scene appearance or light configuration changed.

relaxed (not in the brief): decorative background, exact source-marker hue,
and single-image inference of emitter dimensions/intensity equality/prepass or
lifecycle transitions; brief says: "Area shows a rectangular emitter approximation
whose extents visibly affect representative illumination" and "attached-light
shows the source and illumination moving with its parent." Owner 5353 explicitly
assigns numerical/API/prepass facts to deterministic checks. Score >=6 remains.

All 14 primary and 18 attachment-stage outputs receive distinct descriptive
filenames in the ONE managed job. run.json includes the complete filename/label
manifest; freshness validates it against produced metadata and output files.
Judge receives exactly those labels and names, explicitly --aspect functional,
--min-score 6. It retains every score/process result/stdout/stderr, attempts all
32 judgments even after rejection, and only then fails with the full error list.
The host's current nmc_judge.py supports functional mode and passes image basename,
description and source identity to captures_archive.record_judged. No repository
mailer, archive replacement, approval request or hold release is introduced.
The existing page diagnostic collectors remain active for all pairs.

Command-check now depends on functional checks, and cargo/milestone checks follow
command-check, independently of image judgment. All remain mandatory. This
prevents a rejected image from silently suppressing branch checks. The local
m2-lights-judge-local entry exercises selection, isolated clean, controlled managed
transport, freshness and all-image judgment with CPU fixtures; it starts no
browser and supplies no GPU proof. Its build dependency satisfies the commands
validator. No timeout increased and no foreign milestone command was added.

Instruction audit: read complete execution 13286 checks_json; owner 5353 and
propagations 5354–5358; current milestone 7326/7327 contract; descriptions and
metadata of 7332/7334/7335; dependency edges; and 5349's complete stored work and
verification results. Host functional mode and archive identity support already
exist. 7335 has judge_aspect=functional; 7332/7334 lack that metadata. Explicit
functional mode in the shared command now covers all three consumers. 7332's
embedded old contracts still name capture.png and omit --aspect functional;
7334 names its old producer. Historical supervisor rationale 5349 calls the
background required; 5353 expressly supersedes it. These database descriptions
and historical records were not rewritten: this operator may only GET the API
(and PATCH sibling_repos/env_paths), and may never write the database. The
supervisor must propagate the controlling filename/functional-label contract
into those live card descriptions/metadata without releasing any hold. M3,
archive and successful informational delivery gates remain required.

Evidence boundary: the available archived 13286 render directory retains only
18 PNGs (a partial archive), and its judge.json has zero accepted images. The
live 7335 output retains JSON but no PNGs; 7335-opverify was removed. These are
historical diagnostics only, never replayed proof. No claim of 32 fresh real
images, green GPU flow, archive delivery or email delivery is made by this repair.
The supervisor's preflight and single managed programmatic job at the committed
revision must supply that evidence before one normal card run is allowed.

Local validation before commit: 65 Node tests passed, cargo check exited 0,
and nmc-commands-check accepted all 11 entries. The registered local preflight
is rerun after committing; these CPU checks neither render nor judge real GPU
images. scripts/check.sh remains mandatory in the programmatic command chain.
