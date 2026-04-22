#!/usr/bin/env python3
"""Apply V9 wide-load additions to hk_mla_buffer_managers.cuh.

Idempotent: checks for existing function names before inserting.
"""
import sys
import re

if len(sys.argv) != 3:
    print("usage: apply_buffer_managers_patch.py <target.cuh> <patch.cuh>")
    sys.exit(1)

target_path, patch_path = sys.argv[1], sys.argv[2]

with open(target_path, 'r', encoding='utf-8') as f:
    src = f.read()

with open(patch_path, 'r', encoding='utf-8') as f:
    patch = f.read()

if 'load_k_wide_to_gpr' in src and 'lds_2_gpr_wide' in src:
    print("both patch functions already present, skipping")
    sys.exit(0)

# Extract two function bodies from the patch file.
def extract_between(text, start_marker, end_marker):
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"cannot find start marker: {start_marker}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"cannot find end marker after {start_marker}")
    return text[start:end].strip()

# Pull out the two function bodies.
# Function A: load_k_wide_to_gpr (inside KvManagerV2)
# The patch has separators; extract between known anchors.
section_a_marker = "// === PATCH_SECTION_A_KVMANAGERV2_LOAD_K_WIDE ==="
section_b_marker = "// === PATCH_SECTION_B_QMANAGERV4_LDS_2_GPR_WIDE ==="
section_a_pos = patch.find(section_a_marker)
section_b_pos = patch.find(section_b_marker)
if section_a_pos < 0 or section_b_pos < 0:
    raise RuntimeError("patch section markers not found")

# Body A = content AFTER section_a_marker's header comments, BEFORE section B.
# Walk past the section header (marker line + comment line) until we find the
# first line starting with 4 spaces of C++ code.
def extract_body(patch_text, start_of_section, end_of_section):
    # skip marker line (has // === ...), then skip any // comments at 0-indent.
    lines = patch_text[start_of_section:end_of_section].splitlines()
    body_lines = []
    started = False
    for ln in lines:
        if not started:
            # Start when we see a line that isn't a marker (===) and isn't a top-level comment.
            if ln.startswith("    "):
                started = True
        if started:
            body_lines.append(ln)
    return "\n".join(body_lines).rstrip()

fn_a = extract_body(patch, section_a_pos, section_b_pos)
fn_b = extract_body(patch, section_b_pos, len(patch))

# --- Find insertion point A: KvManagerV2 right after load_k_to_gpr ---
# load_k_to_gpr ends at the closing `}` of its body. Use a more specific anchor.
# Look for "// Load un-transposed vector from LDS to GPR." that is NOT commented twice,
# i.e., a line that begins with "    //" not "        //".
kv2_start = src.find("class KvManagerV2")
if kv2_start < 0:
    raise RuntimeError("cannot find KvManagerV2 class")
# Find the actual docstring (4 spaces indent), not commented-out body text (8 spaces).
import re as _re
marker_a_re = _re.compile(r"^    // Load un-transposed vector from LDS to GPR\.$", _re.MULTILINE)
m = marker_a_re.search(src, kv2_start)
if m is None:
    raise RuntimeError("cannot find insertion anchor A inside KvManagerV2 (4-space-indent docstring)")
marker_a_pos = m.start()

# But we want to insert AFTER the public: label that comes before load_k_to_gpr's section,
# to guarantee the new function is in public scope.
# Actually load_k_to_gpr is already public, so inserting before load_v_to_gpr keeps same scope.

# --- Find insertion point B: QManagerV4 public section after load_q_to_gpr ---
# QManagerV4 has lds_2_gpr in protected section. We need lds_2_gpr_wide in public.
# Insert right before the closing }; of the class body.
q4_start = src.find("class QManagerV4")
if q4_start < 0:
    raise RuntimeError("cannot find QManagerV4 class")
# Find the NEXT class after QManagerV4 — the closing is just before it.
q5_start = src.find("class QManagerV5", q4_start)
if q5_start < 0:
    raise RuntimeError("cannot find QManagerV5 (used to locate V4 end)")
# Walk back from q5_start to find "};" that closes V4.
v4_close = src.rfind("};", q4_start, q5_start)
if v4_close < 0:
    raise RuntimeError("cannot find };  closing QManagerV4")
marker_b_pos = v4_close

# Insertions must proceed back-to-front (LATER position first) so earlier
# positions remain valid in the mutated string. In this file marker_a
# (KvManagerV2 line ~986) is AFTER marker_b (QManagerV4 end line ~500).
if marker_a_pos > marker_b_pos:
    new_src = src[:marker_a_pos] + fn_a + "\n\n" + src[marker_a_pos:]
    new_src = new_src[:marker_b_pos] + fn_b + "\n\n" + new_src[marker_b_pos:]
else:
    new_src = src[:marker_b_pos] + fn_b + "\n\n" + src[marker_b_pos:]
    new_src = new_src[:marker_a_pos] + fn_a + "\n\n" + new_src[marker_a_pos:]

with open(target_path, 'w', encoding='utf-8') as f:
    f.write(new_src)

print(f"inserted load_k_wide_to_gpr + lds_2_gpr_wide into {target_path}")
