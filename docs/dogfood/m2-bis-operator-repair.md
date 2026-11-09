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
