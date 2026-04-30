#!/usr/bin/env python3
"""
Robust extraction: Track brace depth to find function bodies.
"""

with open('src/kernels/cpu/mod.rs', 'r') as f:
    lines = f.readlines()

funcs = []
brace_depth = 0
in_function = False
function_start = None
function_name = None

for i, line in enumerate(lines):
    # Track brace depth globally
    brace_depth += line.count('{')
    brace_depth -= line.count('}')

    # Skip attributes
    stripped = line.strip()
    if stripped.startswith('#'):
        continue

    # If we're not currently in a function, look for fn declaration
    if not in_function:
        if 'fn ' in line:
            # Extract name: pattern could be "fn name(" or "pub fn name(" or "unsafe fn name("
            parts = line.split('fn ', 1)
            if len(parts) == 2:
                after = parts[1].strip()
                name = after.split('(')[0].strip()
                # Validate name
                if name and name.isidentifier() and not any(c in name for c in '{}[]=;'):
                    function_name = name
                    function_start = i
                    in_function = True
    else:
        # We're in a function body; check if brace_depth returned to 0 at this line
        if brace_depth == 0:
            funcs.append((function_name, function_start, i))
            in_function = False
            function_name = None
            function_start = None

print(f"Found {len(funcs)} functions")

# Show all for manual verification
for name, start, end in sorted(funcs, key=lambda x: x[1]):
    print(f"  {name}: lines {start+1}-{end+1}")
