#!/usr/bin/env bash
# Asset guard: fails when a tracked (or, with --staged, a staged) file has a blocked
# extension or is larger than 1 MB. Prints each offender.
# Usage: scripts/check-no-assets.sh [--staged]
set -euo pipefail

MAX_BYTES=$((1024 * 1024))
BLOCKED_RE='\.(glb|gltf|png|jpg|jpeg|avif|ktx2|bin|wasm|hdr|exr)$'

mode=all
case "${1:-}" in
  "") ;;
  --staged) mode=staged ;;
  *) echo "usage: scripts/check-no-assets.sh [--staged]" >&2; exit 2 ;;
esac

cd "$(git rev-parse --show-toplevel)"

# Collect the paths to inspect: the whole index, or only files added/copied/modified/renamed
# in the staged change set.
paths=()
if [[ $mode == staged ]]; then
  while IFS= read -r -d '' p; do paths+=("$p"); done \
    < <(git diff --cached --name-only -z --diff-filter=ACMR)
else
  while IFS= read -r -d '' p; do paths+=("$p"); done < <(git ls-files -z)
fi

offenders=0
for p in "${paths[@]}"; do
  if [[ ${p,,} =~ $BLOCKED_RE ]]; then
    echo "check-no-assets: blocked asset extension: $p" >&2
    offenders=$((offenders + 1))
    continue
  fi
  # Size of the blob in the index (what would be committed), not the working tree copy.
  # Gitlinks (submodules) have no blob and are skipped.
  size=$(git cat-file -s ":$p" 2>/dev/null || echo 0)
  if ((size > MAX_BYTES)); then
    echo "check-no-assets: file larger than 1 MB ($size bytes): $p" >&2
    offenders=$((offenders + 1))
  fi
done

if ((offenders > 0)); then
  echo "check-no-assets: FAIL, $offenders offending file(s); assets never go to git" >&2
  exit 1
fi
echo "check-no-assets: ok (${#paths[@]} $mode file(s) inspected)"
