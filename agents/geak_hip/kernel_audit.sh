#!/bin/bash
# kernel_audit.sh — fail-closed audit on a generated kernel .so / .cu pair.
# Usage:
#   kernel_audit.sh <path-to-.so> [<path-to-.cu>]
# Exit codes:
#   0  — audit passed
#   2  — forbidden token in source
#   3  — forbidden token in disassembly
#   4  — required CDNA4 primitive missing from disassembly
#   5  — file not found / llvm-objdump not available

set -u
SO=${1:?usage: kernel_audit.sh <so> [<cu>]}
CU=${2:-}

ALLOWLIST=${ALLOWLIST:-/tmp/agents/primitives.allowlist}
FORBIDDEN=${FORBIDDEN:-/tmp/agents/forbidden.tokens}

[ -f "$SO" ]        || { echo "AUDIT_FAIL: $SO not found"; exit 5; }
[ -f "$ALLOWLIST" ] || { echo "AUDIT_FAIL: allowlist $ALLOWLIST not found"; exit 5; }
[ -f "$FORBIDDEN" ] || { echo "AUDIT_FAIL: forbidden $FORBIDDEN not found"; exit 5; }
command -v llvm-objdump >/dev/null || { echo "AUDIT_FAIL: llvm-objdump not in PATH"; exit 5; }

echo "[audit] target: $SO"

# 1. Forbidden tokens in source
if [ -n "$CU" ] && [ -f "$CU" ]; then
  while IFS= read -r tok; do
    [ -z "$tok" ] && continue
    [[ "$tok" =~ ^# ]] && continue
    if grep -nF -- "$tok" "$CU" >/dev/null 2>&1; then
      echo "AUDIT_FAIL: forbidden token in source: '$tok' in $CU"
      grep -nF -- "$tok" "$CU" | head -3
      exit 2
    fi
  done < "$FORBIDDEN"
  echo "[audit] source: no forbidden tokens"
fi

# 2. Forbidden tokens in disassembly
DISASM=$(llvm-objdump -d --no-show-raw-insn "$SO" 2>/dev/null)
while IFS= read -r tok; do
  [ -z "$tok" ] && continue
  [[ "$tok" =~ ^# ]] && continue
  if echo "$DISASM" | grep -qF -- "$tok"; then
    echo "AUDIT_FAIL: forbidden token in disassembly: '$tok'"
    exit 3
  fi
done < "$FORBIDDEN"
echo "[audit] disassembly: no forbidden tokens"

# 3. Required CDNA4 primitives in disassembly
declare -i required_found=0
declare -i required_total=0
while IFS= read -r prim; do
  [ -z "$prim" ] && continue
  [[ "$prim" =~ ^# ]] && continue
  required_total+=1
  if echo "$DISASM" | grep -q -- "$prim"; then
    required_found+=1
  else
    echo "[audit] WARN: required primitive missing: $prim"
  fi
done < "$ALLOWLIST"

# Require at least 3 of the 5 mandatory CDNA4 primitives to be present.
# Fewer than 3 means the kernel is falling back to scalar / generic paths
# and is not exploiting CDNA4. Hard fail.
if [ "$required_found" -lt 3 ]; then
  echo "AUDIT_FAIL: only $required_found/$required_total CDNA4 primitives found (need >=3)"
  exit 4
fi

echo "[audit] CDNA4 primitives: $required_found/$required_total present"
echo "[audit] PASS: $SO"
exit 0
