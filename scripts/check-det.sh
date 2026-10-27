#!/usr/bin/env bash
# Deterministic source guard (Q16): scans the Rust sources of the contracted
# sim-path crates for forbidden math, f64, unordered collections, threading,
# clocks and random sources. raa-det is required; raa-anim and raa-anim-graph
# join as soon as those crates exist. Hits print as `file:line: category: code`.
# Exit 0 = clean (explicit zero-hit line), 1 = hits, 2 = scan failure.
# The tick-seconds helpers in crates/raa-det/src/tick.rs are the one sanctioned
# f64 boundary: only TICK_SECONDS_F64 / tick_seconds_f64 lines are exempt.
# Usage: scripts/check-det.sh [--self-test]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
F64_BOUNDARY=crates/raa-det/src/tick.rs
scan_error() {
  printf 'check-det: SCAN ERROR: %s\n' "$*" >&2
  exit 2
}
# Prints one line per hit: file:line: category: code.
scan_file() {
  awk -v file="$1" -v f64_allow="$F64_BOUNDARY" '
    BEGIN {
      in_block = 0; in_str = 0; esc = 0
      ncats = 6
      cats[1] = "math"
      pat[1] = "(^|[^A-Za-z0-9_])(sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|asinh|acosh|atanh|exp|exp2|exp_m1|expm1|ln|ln_1p|log|log2|log10|log1p|pow|powf|powi|cbrt|hypot|fma|mul_add|to_degrees|to_radians)(f|f32|f64)?[[:space:]]*\\("
      cats[2] = "f64"
      pat[2] = "(^|[^A-Za-z0-9_])f64([^A-Za-z0-9_]|$)"
      cats[3] = "collections"
      pat[3] = "(^|[^A-Za-z0-9_])(HashMap|HashSet|hash_map|hash_set|hashbrown)([^A-Za-z0-9_]|$)"
      cats[4] = "threading"
      pat[4] = "(^|[^A-Za-z0-9_])(thread|thread_local|Thread|JoinHandle|Mutex|RwLock|Condvar|Barrier|OnceLock|OnceCell|LazyLock|lazy_static|rayon|crossbeam|mpsc|Atomic[A-Za-z0-9_]*|std::sync|Arc)([^A-Za-z0-9_]|$)"
      cats[5] = "clock"
      pat[5] = "(^|[^A-Za-z0-9_])(Instant|SystemTime|UNIX_EPOCH|chrono|SystemClock|std::time|time::)([^A-Za-z0-9_]|$)"
      cats[6] = "random"
      pat[6] = "(^|[^A-Za-z0-9_])(rand|rand_core|rand_chacha|fastrand|nanorand|getrandom|OsRng|ThreadRng|StdRng|SmallRng|SeedableRng|thread_rng|random|Random)([^A-Za-z0-9_]|$)"
    }
    function strip(s,    out, i, n, c) {
      out = ""; i = 1; n = length(s)
      while (i <= n) {
        c = substr(s, i, 1)
        if (in_block) {
          if (c == "*" && substr(s, i + 1, 1) == "/") { in_block = 0; i += 2 } else { i++ }
          continue
        }
        if (in_str) {
          if (esc) { esc = 0 }
          else if (c == "\\") { esc = 1 }
          else if (c == "\"") { in_str = 0; out = out c }
          i++
          continue
        }
        if (c == "\"") { in_str = 1; out = out c; i++; continue }
        if (c == "/" && substr(s, i + 1, 1) == "*") { in_block = 1; i += 2; continue }
        if (c == "/" && substr(s, i + 1, 1) == "/") break
        out = out c
        i++
      }
      return out
    }
    {
      code = strip($0)
      if (code ~ /^[[:space:]]*$/) next
      for (k = 1; k <= ncats; k++) {
        if (code !~ pat[k]) continue
        if (cats[k] == "f64" && file == f64_allow &&
            code ~ /TICK_SECONDS_F64|tick_seconds_f64|ANIM_TICK_HZ[[:space:]]+as[[:space:]]+f64/) continue
        printf "%s:%d: %s: %s\n", file, NR, cats[k], code
      }
    }
  ' "$1"
}

self_test() {
  local tmp out cat
  tmp=$(mktemp -d) || scan_error "mktemp failed"
  trap "rm -rf '$tmp'" EXIT
  cat >"$tmp/positive.rs" <<'EOF'
use std::{collections::HashMap, time::Instant};
fn math(x: f64) -> f64 { x.sin() }
fn threads() { std::thread::spawn(|| {}); }
fn clock() -> Instant { Instant::now() }
fn rng() -> u32 { rand::random() }
EOF
  cat >"$tmp/negative.rs" <<'EOF'
// f64 HashMap Instant thread sin( rand::random()
const RATE: f32 = 1.0 / 30.0;
fn advance(p: [f32; 3]) -> [f32; 3] {
    let inverse = 1.0 / (p[0] * p[0] + p[1] * p[1]).sqrt();
    [p[0] * inverse, p[1] * inverse, RATE]
}
EOF
  out=$(scan_file "$tmp/positive.rs")
  for cat in math f64 collections threading clock random; do
    if ! grep -q ": $cat:" <<<"$out"; then
      printf 'check-det: self-test FAIL: positive sample escaped the %s rule\n' "$cat" >&2
      return 1
    fi
  done
  out=$(scan_file "$tmp/negative.rs")
  if [[ -n $out ]]; then
    printf 'check-det: self-test FAIL: negative sample produced hits:\n%s\n' "$out" >&2
    return 1
  fi
  echo "check-det: self-test ok (6 categories detected; clean sample reports 0 hits)"
}

case "${1:-}" in
  "") ;;
  --self-test) self_test; exit $? ;;
  *) scan_error "usage: scripts/check-det.sh [--self-test]" ;;
esac

crates=(raa-det)
[[ -d crates/raa-anim ]] && crates+=(raa-anim)
[[ -d crates/raa-anim-graph ]] && crates+=(raa-anim-graph)

[[ -d crates/raa-det/src ]] || scan_error "required raa-det source directory is missing"

files=()
for crate in "${crates[@]}"; do
  [[ -d crates/$crate ]] || scan_error "contracted crate crates/$crate is missing"
  found=0
  while IFS= read -r f; do
    [[ -r $f ]] || scan_error "cannot read $f"
    files+=("$f")
    found=1
  done < <(find "crates/$crate" -type f -name '*.rs' -not -path '*/target/*' -print | LC_ALL=C sort)
  ((found)) || scan_error "crates/$crate contains no Rust sources"
done
((${#files[@]})) || scan_error "no Rust sources found"

hits=0
declare -A hit_files=()
for f in "${files[@]}"; do
  out=$(scan_file "$f") || scan_error "scanner failed on $f"
  [[ -n $out ]] || continue
  printf '%s\n' "$out"
  hits=$((hits + $(grep -c . <<<"$out")))
  hit_files[$f]=1
done

if ((hits > 0)); then
  printf 'check-det: FAIL: %d hit(s) in %d file(s); determinism forbids these sources\n' \
    "$hits" "${#hit_files[@]}" >&2
  exit 1
fi
printf 'check-det: 0 hits across %d Rust file(s) in %s\n' \
  "${#files[@]}" "$(IFS=', '; echo "${crates[*]}")"
