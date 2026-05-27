#!/usr/bin/env python3
"""
Duplicate code finder script.
Usage: python .find_duplicates.py <directory> [--min-lines 5] [--min-similarity 0.85] [--exclude "*.pyc,__pycache__,.git,target"]

Finds duplicate and near-duplicate code blocks in source files.
Reports exact duplicates (same lines) and structural duplicates (similar function/class bodies).
"""
import os
import sys
import re
import hashlib
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
from difflib import SequenceMatcher

# Languages and their comment syntax
COMMENT_PATTERNS = {
    '.py': (r'#.*$', r'""".*?"""', r"'''.*?'''"),
    '.rs': (r'//.*$', r'/\*.*?\*/'),
    '.toml': (r'#.*$',),
    '.md': (),
    '.sh': (r'#.*$',),
    '.yml': (r'#.*$',),
    '.yaml': (r'#.*$',),
}

EXCLUDE_DIRS = {'__pycache__', '.git', '.cargo', 'target', '.venv', '.venv_test', '.pytest_cache', '.ruff_cache', '.benchmarks', '.kilo', '__pycache__'}
EXCLUDE_EXTS = {'.pyc', '.pyo', '.o', '.so', '.dylib', '.dll', '.exe', '.mod', '.sum', '.lock', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.svg', '.whl', '.egg-info'}

def get_comment_regex(ext: str):
    """Get regex patterns for comments in the given file extension."""
    patterns = COMMENT_PATTERNS.get(ext, (r'//.*$', r'/\*.*?\*/'))
    combined = '|'.join(patterns)
    return re.compile(combined, re.DOTALL | re.MULTILINE) if '"""' in combined or "'''" in combined or '/*' in combined else re.compile(combined, re.MULTILINE)

def strip_comments(text: str, ext: str) -> str:
    """Strip comments from code."""
    regex = get_comment_regex(ext)
    return regex.sub('', text)

def normalize_line(line: str) -> str:
    """Normalize a line for comparison: strip whitespace, collapse spaces."""
    line = line.strip()
    line = re.sub(r'\s+', ' ', line)
    return line

def chunk_file(filepath: str, min_lines: int = 5) -> List[Tuple[int, int, List[str]]]:
    """
    Split a file into logical code chunks (functions, classes, top-level blocks).
    Returns list of (start_line, end_line, normalized_lines).
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Strip comments
    clean = strip_comments(content, ext)
    lines = clean.split('\n')
    raw_lines = content.split('\n')
    
    chunks = []
    
    if ext == '.py':
        # Python: find functions and classes
        pattern = re.compile(r'^(class |async def |def |@\w)', re.MULTILINE)
        matches = list(pattern.finditer(clean))
        for i, match in enumerate(matches):
            start = clean[:match.start()].count('\n')
            if i + 1 < len(matches):
                end = clean[:matches[i+1].start()].count('\n') - 1
            else:
                end = len(lines) - 1
            
            chunk_lines = [normalize_line(l) for l in lines[start:end+1] if normalize_line(l)]
            chunk_lines = [l for l in chunk_lines if l and not l.startswith(('import ', 'from ', '#', '"""', "'''"))]
            if len(chunk_lines) >= min_lines:
                chunks.append((start + 1, end + 1, chunk_lines))
    
    elif ext == '.rs':
        # Rust: find fn, struct, impl, trait blocks
        pattern = re.compile(r'^\s*(pub\s+)?(fn |struct |enum |impl|trait |mod |macro_rules!|unsafe\s+fn)', re.MULTILINE)
        matches = list(pattern.finditer(clean))
        for i, match in enumerate(matches):
            start = clean[:match.start()].count('\n')
            if i + 1 < len(matches):
                end = clean[:matches[i+1].start()].count('\n') - 1
            else:
                end = len(lines) - 1
            
            chunk_lines = [normalize_line(l) for l in lines[start:end+1] if normalize_line(l)]
            chunk_lines = [l for l in chunk_lines if l and not l.startswith(('use ', '//', '/*'))]
            if len(chunk_lines) >= min_lines:
                chunks.append((start + 1, end + 1, chunk_lines))
    
    # Also add sliding window blocks for any file type (catches non-structural duplicates)
    # We'll do this in the comparison phase instead to avoid O(n^2) blowup
    
    return chunks

def find_exact_duplicates(files: List[str], min_lines: int = 5) -> List[Dict]:
    """
    Find exact duplicate blocks across files using line hashing.
    """
    # Hash all file lines
    file_lines: Dict[str, List[str]] = {}
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            clean = strip_comments(content, ext)
            lines = [normalize_line(l) for l in clean.split('\n')]
            file_lines[fp] = lines
        except:
            continue
    
    # Find duplicated sequences using rolling hash approach
    # For each file, find blocks of N+ consecutive lines that appear elsewhere
    duplicates = []
    
    # Build hash map: line_content -> list of (file, line_num)
    line_map: Dict[str, List[Tuple[str, int]]] = {}
    for fp, lines in file_lines.items():
        for i, line in enumerate(lines):
            if line:  # skip empty lines
                if line not in line_map:
                    line_map[line] = []
                line_map[line].append((fp, i + 1))
    
    # Find consecutive duplicate sequences
    visited: Set[Tuple[str, int]] = set()
    
    for fp, lines in file_lines.items():
        i = 0
        while i < len(lines):
            if not lines[i] or (fp, i) in visited:
                i += 1
                continue
            
            # Try to extend a match
            candidates = [(f, ln) for f, ln in line_map.get(lines[i], []) if f != fp or abs(ln - (i+1)) > 2]
            if not candidates:
                i += 1
                continue
            
            best_matches = []
            for cf, cln in candidates:
                # Count how many consecutive lines match
                offset = 0
                while (i + offset < len(lines) and 
                       cln + offset - 1 < len(file_lines[cf]) and
                       lines[i + offset] and 
                       file_lines[cf][cln + offset - 1] and
                       lines[i + offset] == file_lines[cf][cln + offset - 1]):
                    offset += 1
                
                if offset >= min_lines:
                    locations = [(fp, i + 1, i + offset), (cf, cln, cln + offset - 1)]
                    best_matches.append((offset, locations))
            
            if best_matches:
                best_matches.sort(key=lambda x: -x[0])
                offset, locations = best_matches[0]
                
                # Mark visited
                for loc_fp, loc_start, loc_end in locations:
                    for ln in range(loc_start - 1, loc_end):
                        visited.add((loc_fp, ln))
                
                # Get the actual code
                code_lines = file_lines[locations[0][0]][locations[0][1]-1:locations[0][2]]
                
                duplicates.append({
                    'type': 'exact',
                    'length': offset,
                    'locations': locations,
                    'lines': code_lines[:10] + (['...'] if offset > 10 else []),
                })
                i += offset
            else:
                i += 1
    
    return duplicates

def find_near_duplicates(files: List[str], min_lines: int = 5, min_similarity: float = 0.85) -> List[Dict]:
    """
    Find similar (near-duplicate) code blocks using difflib.
    Compares all pairs of chunks within the same file group.
    """
    # Get all code chunks
    all_chunks: List[Tuple[str, int, int, List[str]]] = []
    for fp in files:
        chunks = chunk_file(fp, min_lines)
        for start, end, chunk_lines in chunks:
            all_chunks.append((fp, start, end, chunk_lines))
    
    near_dups = []
    compared: Set[Tuple[int, int]] = set()
    
    for i in range(len(all_chunks)):
        for j in range(i + 1, len(all_chunks)):
            if (i, j) in compared:
                continue
            compared.add((i, j))
            
            fp1, s1, e1, lines1 = all_chunks[i]
            fp2, s2, e2, lines2 = all_chunks[j]
            
            # Skip if same file and overlapping ranges
            if fp1 == fp2 and (s1 <= e2 and s2 <= e1):
                continue
            
            # Compare using SequenceMatcher
            matcher = SequenceMatcher(None, lines1, lines2)
            similarity = matcher.ratio()
            
            if similarity >= min_similarity:
                near_dups.append({
                    'type': 'near-duplicate',
                    'similarity': round(similarity, 3),
                    'locations': [
                        (fp1, s1, e1),
                        (fp2, s2, e2),
                    ],
                    'lines': lines1[:8] + (['...'] if len(lines1) > 8 else []),
                })
    
    return near_dups

def find_repeated_patterns(files: List[str], min_lines: int = 5) -> List[Dict]:
    """
    Find patterns that repeat within the same file (copy-paste within a file).
    """
    duplicates = []
    
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            clean = strip_comments(content, ext)
            lines = [normalize_line(l) for l in clean.split('\n')]
        except:
            continue
        
        # Look for repeated blocks within the same file
        for start in range(len(lines)):
            if not lines[start]:
                continue
            
            for end in range(start + min_lines, min(start + 50, len(lines) + 1)):
                block = '\n'.join(lines[start:end])
                if not block.strip():
                    continue
                
                # Search for this block elsewhere in the file
                block_len = end - start
                search_from = start + 1
                
                while search_from < len(lines):
                    # Find the first matching line
                    found = -1
                    for k in range(search_from, len(lines)):
                        if k + block_len <= len(lines) and lines[k] == lines[start]:
                            found = k
                            break
                    
                    if found == -1 or found + block_len > len(lines):
                        break
                    
                    # Check if the full block matches
                    match = True
                    for k in range(block_len):
                        if lines[start + k] != lines[found + k]:
                            match = False
                            break
                    
                    if match and found - start >= min_lines:
                        duplicates.append({
                            'type': 'intra-file',
                            'length': block_len,
                            'locations': [
                                (fp, start + 1, end),
                                (fp, found + 1, found + block_len),
                            ],
                            'lines': lines[start:start+min(8, block_len)] + (['...'] if block_len > 8 else []),
                        })
                        search_from = found + block_len
                    else:
                        search_from = found + 1
    
    return duplicates

def find_duplicates_in_directory(directory: str, min_lines: int = 5, min_similarity: float = 0.85) -> Dict:
    """Main function: find all types of duplicates in a directory."""
    dir_path = Path(directory).resolve()
    
    # Collect all source files
    files = []
    for root, dirs, filenames in os.walk(dir_path):
        # Skip excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        
        for fn in filenames:
            fp = os.path.join(root, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in EXCLUDE_EXTS:
                continue
            if fn.endswith(('.pyc',)):
                continue
            # Only source-like files
            if ext in ('.py', '.rs', '.toml', '.sh', '.md', '.yml', '.yaml', '.json', '.js', '.ts', '.c', '.cpp', '.h', '.hpp', '.go', '.java', '.rs'):
                files.append(fp)
    
    print(f"Scanning {len(files)} files in {directory}", file=sys.stderr)
    
    results = {
        'directory': directory,
        'files_scanned': len(files),
        'exact': [],
        'near_duplicate': [],
        'intra_file': [],
    }
    
    # Find exact duplicates (fast, hash-based)
    if files:
        results['exact'] = find_exact_duplicates(files, min_lines)
    
    # Find near duplicates (slower, uses difflib)
    if files and len(files) <= 200:  # limit to avoid O(n²) blowup
        results['near_duplicate'] = find_near_duplicates(files, min_lines, min_similarity)
    
    # Find intra-file repeats
    if files:
        results['intra_file'] = find_repeated_patterns(files, min_lines)
    
    return results

def print_results(results: Dict):
    """Pretty-print duplicate findings."""
    directory = results['directory']
    print(f"\n{'='*80}")
    print(f"DUPLICATE CODE REPORT: {directory}")
    print(f"Files scanned: {results['files_scanned']}")
    print(f"{'='*80}")
    
    categories = [
        ('exact', 'EXACT DUPLICATE BLOCKS'),
        ('near_duplicate', 'NEAR-DUPLICATE (SIMILAR) BLOCKS'),
        ('intra_file', 'INTRA-FILE REPEATED PATTERNS'),
    ]
    
    total = 0
    for key, title in categories:
        items = results[key]
        if items:
            # Deduplicate by grouping similar locations
            seen_locs = set()
            unique_items = []
            for item in items:
                locs = tuple(sorted(item['loc'] if 'loc' in item else str(item['locations'])))
                if locs not in seen_locs:
                    seen_locs.add(locs)
                    unique_items.append(item)
            
            print(f"\n{'─'*80}")
            print(f"🔍 {title}: {len(unique_items)} found")
            print(f"{'─'*80}")
            total += len(unique_items)
            
            for idx, dup in enumerate(unique_items[:50]):  # Limit output
                print(f"\n  #{idx+1} [{dup['type'].upper()}] ", end='')
                if 'similarity' in dup:
                    print(f"Similarity: {dup['similarity']*100:.1f}%", end='')
                if 'length' in dup:
                    print(f" | {dup['length']} lines", end='')
                print()
                
                for loc in dup['locations']:
                    if len(loc) == 3:
                        fp, start, end = loc
                        rel = os.path.relpath(fp, startswith=results.get('directory', ''))
                        print(f"      📄 {fp}:{start}-{end}")
                    elif len(loc) == 2:
                        fp, ln = loc
                        print(f"      📄 {fp}:{ln}")
                
                if dup.get('lines'):
                    print(f"      Code:")
                    for l in dup['lines'][:5]:
                        print(f"        │ {l}")
                    if 'similarity' in dup and dup['similarity'] < 1.0:
                        print(f"        │ (showing first file's version)")
                
                if idx >= 49:
                    print(f"\n  ... and {len(unique_items) - 50} more")
                    break
            
            if not unique_items:
                print("  (none found)")
        else:
            print(f"\n  {title}: 0 found")
    
    print(f"\n{'='*80}")
    print(f"TOTAL duplicate groups found: {total}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Find duplicate code')
    parser.add_argument('directory', help='Directory to scan')
    parser.add_argument('--min-lines', type=int, default=5, help='Minimum lines for a duplicate block')
    parser.add_argument('--min-similarity', type=float, default=0.85, help='Minimum similarity ratio for near-duplicates')
    parser.add_argument('--output', '-o', help='Output file (JSON)')
    
    args = parser.parse_args()
    
    results = find_duplicates_in_directory(args.directory, args.min_lines, args.min_similarity)
    
    if args.output:
        import json
        # Make locations serializable
        serializable = []
        for key in ['exact', 'near_duplicate', 'intra_file']:
            for item in results[key]:
                s_item = dict(item)
                s_item['locations'] = [[str(x) for x in loc] for loc in s_item['locations']]
                serializable.append(s_item)
        
        with open(args.output, 'w') as f:
            json.dump({'directory': args.directory, 'files_scanned': results['files_scanned'], 'duplicates': serializable}, f, indent=2)
        print(f"\nResults written to {args.output}", file=sys.stderr)
    
    print_results(results)

if __name__ == '__main__':
    main()
