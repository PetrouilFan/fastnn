#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

baseline_file="ci/rustfmt-baseline.txt"
if [[ ! -f "$baseline_file" ]]; then
  echo "missing baseline file: $baseline_file" >&2
  exit 1
fi

mapfile -t rust_files < <(git ls-files '*.rs')
if [[ ${#rust_files[@]} -eq 0 ]]; then
  echo "No Rust files tracked by git."
  exit 0
fi

actual=$(mktemp)
baseline=$(mktemp)
trap 'rm -f "$actual" "$baseline"' EXIT

for file in "${rust_files[@]}"; do
  if ! rustfmt --check --edition 2021 "$file" >/dev/null 2>&1; then
    printf '%s\n' "$file" >> "$actual"
  fi
done

sort -u "$actual" -o "$actual"
grep -vE '^\s*(#|$)' "$baseline_file" | sort -u > "$baseline"

new_debt=$(comm -13 "$baseline" "$actual")
retired_debt=$(comm -23 "$baseline" "$actual")

if [[ -n "$retired_debt" ]]; then
  echo "rustfmt baseline entries no longer needed:"
  printf '  %s\n' $retired_debt
fi

if [[ -n "$new_debt" ]]; then
  echo "New rustfmt debt detected outside the baseline:" >&2
  printf '  %s\n' $new_debt >&2
  exit 1
fi

count=$(wc -l < "$actual" | tr -d ' ')
echo "rustfmt baseline check passed ($count tracked debt files, no new ones)."
