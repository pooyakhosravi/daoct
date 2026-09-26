"""Verify the original submission snapshot without runtime dependencies."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "submission"
    manifest = json.loads((root / "docs/submission_manifest.json").read_text())
    expected = manifest["files"]
    actual = {
        p.relative_to(source).as_posix()
        for p in source.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if actual != set(expected):
        raise RuntimeError(f"Missing: {set(expected) - actual}; extra: {actual - set(expected)}")
    for name, info in expected.items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != info["sha256"] or len(data) != info["bytes"]:
            raise RuntimeError(f"Submitted file changed: {name}")
        if name.endswith(".py"):
            tree = ast.parse(data, filename=name)
            # Catch accidentally omitted local packages, especially daoct/data.
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("daoct."):
                    module = source.joinpath(*node.module.split("."))
                    if not module.with_suffix(".py").is_file() and not (module / "__init__.py").is_file():
                        raise RuntimeError(f"Missing local import: {node.module}")
    print(f"PASS: {len(expected)} submitted files match their hashes; Python syntax and local imports checked.")


if __name__ == "__main__":
    main()
