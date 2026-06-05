#!/usr/bin/env python3
"""Scan a directory for all imports from the miles package.

Outputs JSON with import details for cross-referencing with upstream changes.
This is a **secondary signal** for miles-upstream-prs — miles-imp itself IS
miles (we modify the package in place), so the primary signal is "files we've
modified vs merge-base." This scanner is here for forward-compat: if/when a
downstream consumer imports `miles.*` as a library, this lets the skill flag
upstream PRs that touch modules the consumer depends on.

Usage:
    python scan_imports.py <directory-to-scan>

Example:
    python scan_imports.py ./examples/vagen
"""

import json
import re
import sys
from pathlib import Path


def scan_imports(base_path: Path) -> list[dict]:
    """Scan Python files for miles imports.

    Returns list of dicts with:
        - file: relative file path
        - line: line number
        - import_statement: full import line
        - module: the miles module being imported (e.g., miles.utils.types)
        - names: list of specific names imported (e.g., ['Sample'])
    """
    results = []

    from_import_pattern = re.compile(r"^(?:\s*)from\s+(miles(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import\s+(.+)$")
    direct_import_pattern = re.compile(r"^(?:\s*)import\s+(miles(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)(?:\s+as\s+\w+)?$")

    for py_file in base_path.rglob("*.py"):
        try:
            content = py_file.read_text()
        except Exception:
            continue

        for line_num, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            match = from_import_pattern.match(line)
            if match:
                module = match.group(1)
                imports_str = match.group(2)
                names = [
                    name.strip().split(" as ")[0].strip()
                    for name in imports_str.replace("(", "").replace(")", "").split(",")
                    if name.strip() and not name.strip().startswith("#")
                ]
                results.append(
                    {
                        "file": str(py_file.relative_to(base_path)),
                        "line": line_num,
                        "import_statement": stripped,
                        "module": module,
                        "names": names,
                        "type": "from_import",
                    }
                )
                continue

            match = direct_import_pattern.match(line)
            if match:
                module = match.group(1)
                results.append(
                    {
                        "file": str(py_file.relative_to(base_path)),
                        "line": line_num,
                        "import_statement": stripped,
                        "module": module,
                        "names": [],
                        "type": "direct_import",
                    }
                )

    return sorted(results, key=lambda x: (x["file"], x["line"]))


def get_watched_modules(imports: list[dict]) -> dict[str, list[dict]]:
    """Group imports by module for cross-referencing.

    Returns dict mapping module path to list of (file, line, names) consumers.
    """
    modules: dict[str, list[dict]] = {}
    for imp in imports:
        modules.setdefault(imp["module"], []).append({"file": imp["file"], "line": imp["line"], "names": imp["names"]})
    return modules


def main():
    if len(sys.argv) < 2:
        print("Usage: python scan_imports.py <directory-to-scan>")
        print("Example: python scan_imports.py ./examples/vagen")
        sys.exit(1)

    scan_path = Path(sys.argv[1])

    if not scan_path.exists():
        print(json.dumps({"error": f"Directory not found at {scan_path}"}))
        sys.exit(1)

    if not scan_path.is_dir():
        print(json.dumps({"error": f"Path {scan_path} is not a directory"}))
        sys.exit(1)

    imports = scan_imports(scan_path)
    modules = get_watched_modules(imports)

    output = {
        "total_imports": len(imports),
        "unique_modules": len(modules),
        "imports": imports,
        "modules": modules,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
