#!/usr/bin/env python3
"""Phase2 kernel fixups:
1. Add #include "aiter_opus_plus.h" if missing
2. Remove unused `using OP = opus::vector_t<OutT, ...>` at 2stage launcher
"""
PATH = "/app/aiter-test/csrc/include/custom_all_reduce.cuh"
src = open(PATH).read()
ORIG = src

# Fix 1: include
if '#include "aiter_opus_plus.h"' not in src:
    OLD_INC = '#include "opus/opus.hpp"'
    NEW_INC = '#include "opus/opus.hpp"\n#include "aiter_opus_plus.h"'
    assert OLD_INC in src
    src = src.replace(OLD_INC, NEW_INC, 1)
    print("Added #include aiter_opus_plus.h")
else:
    print("aiter_opus_plus.h already included")

# Fix 2: remove unused `using OP` in 2stage launcher (eagerly fails on fp4_t)
OLD_OP = '    using OP                = opus::vector_t<OutT, 16 / sizeof(T)>;\n'
if OLD_OP in src:
    src = src.replace(OLD_OP, '', 1)
    print("Removed unused using OP in 2stage launcher")
else:
    print("OP already removed")

if src != ORIG:
    open(PATH, 'w').write(src)
    print(f"WROTE {PATH}")
