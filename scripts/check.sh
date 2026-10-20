#!/usr/bin/env bash
# Workspace gate (Quality bar Q1). Prints `== <step>` per step and exits non-zero on the
# first failure. Never assumes a member crate exists: on a workspace with zero packages
# the cargo steps are skipped (with a message) and only the non-cargo guards run.
# Usage: scripts/check.sh [--quick]   (--quick skips `cargo test`)
set -euo pipefail

quick=0
case "${1:-}" in
  "") ;;
  --quick) quick=1 ;;
  *) echo "usage: scripts/check.sh [--quick]" >&2; exit 2 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."
shopt -s nullglob

step() { echo "== $*"; }
fail() { echo "check.sh: FAIL: $*" >&2; exit 1; }

# Existing directories among the given ones (the guards never assume a layout exists).
existing() { local d; for d in "$@"; do [[ -e $d ]] && echo "$d"; done; return 0; }

step "cargo metadata"
metadata=$(cargo metadata --no-deps --format-version 1) || fail "cargo metadata failed"
if [[ $metadata == '{"packages":[]'* ]]; then
  has_members=0
  echo "check.sh: workspace has no member crates yet; skipping cargo fmt/check/clippy/test"
else
  has_members=1
fi

if ((has_members)); then
  # Build with the rustc on PATH, whose components are checked below: a cargo config
  # `build.rustc` may name another toolchain without the wasm32 std, and mixing it with
  # the PATH clippy-driver and rustdoc fails with E0514. An explicit RUSTC wins.
  if [[ -z ${RUSTC:-} ]] && command -v rustc >/dev/null; then
    RUSTC=$(command -v rustc)
    export RUSTC
  fi
  step "toolchain components"
  missing=()
  cargo fmt --version >/dev/null 2>&1 || missing+=("rustfmt")
  cargo clippy --version >/dev/null 2>&1 || missing+=("clippy")
  wasm_libdir=$(rustc --print target-libdir --target wasm32-unknown-unknown 2>/dev/null || true)
  [[ -n $wasm_libdir && -d $wasm_libdir ]] || missing+=("rust-std for wasm32-unknown-unknown")
  for m in "${missing[@]}"; do
    case $m in
      rustfmt | clippy) echo "check.sh: missing toolchain component: $m (rustup component add $m)" >&2 ;;
      *) echo "check.sh: missing toolchain component: $m (rustup target add wasm32-unknown-unknown)" >&2 ;;
    esac
  done
  ((${#missing[@]} == 0)) || fail "required toolchain component(s) missing: ${missing[*]}"

  step "cargo fmt --all --check"
  cargo fmt --all --check || fail "cargo fmt"

  step "cargo clippy (host)"
  cargo clippy --workspace --all-targets -- -D warnings || fail "cargo clippy (host)"

  step "cargo clippy (wasm32-unknown-unknown)"
  cargo clippy --workspace --lib --target wasm32-unknown-unknown -- -D warnings \
    || fail "cargo clippy (wasm32)"

  if ((quick)); then
    step "cargo test --workspace (skipped by --quick)"
  else
    step "cargo test --workspace"
    cargo test --workspace || fail "cargo test"
  fi

  step "demo wasm builds"
  demos=(demos/*/)
  if ((${#demos[@]} == 0)); then
    echo "no demo crates"
  fi
  for d in "${demos[@]}"; do
    d=${d%/}
    [[ -f $d/Cargo.toml ]] || continue
    [[ -x scripts/build-web.sh ]] || fail "scripts/build-web.sh is missing or not executable (needed for $d)"
    echo "-- $d"
    scripts/build-web.sh "$d" || fail "build-web $d"
  done
fi

step "asset guard"
scripts/check-no-assets.sh || fail "asset guard"

step "SAFETY guard"
# Every Rust line containing the token `unsafe` (comment lines excepted) needs, on the line
# directly above, `// SAFETY: <reason> (test: NAME)`, and `fn NAME` must exist in the workspace.
mapfile -t rust_dirs < <(existing crates demos tests)
safety_errors=0
if ((${#rust_dirs[@]} > 0)); then
  while IFS= read -r -d '' file; do
    prev=""
    lineno=0
    while IFS= read -r line || [[ -n $line ]]; do
      lineno=$((lineno + 1))
      if [[ $line =~ (^|[^A-Za-z0-9_])unsafe([^A-Za-z0-9_]|$) && ! $line =~ ^[[:space:]]*// ]]; then
        if [[ $prev =~ ^[[:space:]]*//[[:space:]]*SAFETY:.+\(test:[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)\)[[:space:]]*$ ]]; then
          test_fn=${BASH_REMATCH[1]}
          if ! grep -rqE --include='*.rs' "fn[[:space:]]+${test_fn}[[:space:]]*[<(]" "${rust_dirs[@]}"; then
            echo "$file:$lineno: SAFETY comment names test '$test_fn' but no 'fn $test_fn' exists" >&2
            safety_errors=$((safety_errors + 1))
          fi
        else
          echo "$file:$lineno: 'unsafe' without '// SAFETY: <reason> (test: NAME)' directly above" >&2
          safety_errors=$((safety_errors + 1))
        fi
      fi
      prev=$line
    done < "$file"
  done < <(find "${rust_dirs[@]}" -name target -prune -o -type f -name '*.rs' -print0)
fi
((safety_errors == 0)) || fail "SAFETY guard ($safety_errors violation(s))"
echo "SAFETY guard: ok"

step "neutrality guard"
mapfile -t neutral_paths < <(existing crates demos tests packages examples docs/API.md)
if ((${#neutral_paths[@]} > 0)); then
  if grep -rnIiE --exclude-dir=target --exclude-dir=node_modules \
    'royal|fiesta|monster|racer' "${neutral_paths[@]}" >&2; then
    fail "neutrality guard: the public API must not name a consumer game (matches above)"
  fi
fi
echo "neutrality guard: ok"

echo "check.sh: all checks passed"
