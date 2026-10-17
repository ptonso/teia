#!/usr/bin/env bash
# Run every script of one end-to-end pack and print a pass/fail table.
# A run passes when it exits 0 and leaves eval/summary.yaml (sweep: sweep.yaml) (RL, whose eval is off: a checkpoint).
# Output streams live and is also saved to tests/output/<pack>/logs/<name>.log.
# Usage: bash tests/_run_pack.sh <pack> [teia overrides...]   (from the repo root, inside the env that has teia; TEIA=/path/to/teia overrides the binary)
set -uo pipefail
marker=eval/summary.yaml; [[ "$1" == sweep ]] && marker=sweep.yaml

pack="$1"; shift
runs_root="tests/output/${pack}/runs/${pack}-tests"
logs="tests/output/${pack}/logs"
mkdir -p "$logs"
results=()
failed=0
for script in "tests/${pack}-runs/run/"*.sh; do
  name="$(basename "$script" .sh)"
  echo "=== $name ==="
  bash "$script" "$@" 2>&1 | tee "$logs/$name.log"
  code=${PIPESTATUS[0]}
  if [[ $code -eq 0 ]]; then
    run_dir="$(ls -dt "$runs_root/$name"/*/ 2>/dev/null | head -1)"
    if [[ -n "$run_dir" && ( -f "$run_dir/$marker" || ( "$pack" == rl && -n "$(find "$run_dir" -name '*.ckpt' -print -quit)" ) ) ]]; then
      status=PASS
    else
      status="FAIL (no $marker)"
    fi
  else
    status="FAIL (exit $code, see $logs/$name.log)"
  fi
  [[ "$status" == PASS ]] || failed=$((failed + 1))
  results+=("$(printf '%-45s %s' "$name" "$status")")
done

echo
echo "########## ${pack}: $(( ${#results[@]} - failed ))/${#results[@]} passed ##########"
printf '%s\n' "${results[@]}"
exit $(( failed > 0 ))
