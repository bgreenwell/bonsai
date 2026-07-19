"""Run every assets/tests/**/generate.py, optionally filtered by framework.

Usage:
    python scripts/generate_all_fixtures.py [framework ...]

With no arguments all generation scripts run. Passing framework directory
names (e.g. `xgboost lightgbm catboost sklearn_onnx`) restricts the run to
those subdirectories of assets/tests. A script failure does not stop the
run, but the exit code is non-zero if any script failed.
"""

import subprocess
import sys
from pathlib import Path


def main():
    base_dir = Path("assets/tests")
    if not base_dir.exists():
        print(f"Error: {base_dir} not found. Run from project root.")
        sys.exit(1)

    frameworks = set(sys.argv[1:])
    scripts = sorted(base_dir.glob("**/generate.py"))
    if frameworks:
        scripts = [s for s in scripts if s.relative_to(base_dir).parts[0] in frameworks]

    if not scripts:
        print("No generate.py scripts found.")
        sys.exit(1)

    print(f"Found {len(scripts)} generation scripts.")

    failures = []
    for script in scripts:
        print("-" * 60)
        print(f"Running: {script}")
        try:
            subprocess.run(
                [sys.executable, "generate.py"],
                cwd=script.parent,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
            failures.append(script)
        except Exception as e:
            print(f"Unexpected error: {e}")
            failures.append(script)

    write_versions(base_dir)

    print("-" * 60)
    if failures:
        print(f"{len(failures)} script(s) failed:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("Finished processing all generation scripts.")


def write_versions(base_dir: Path):
    """Record which framework versions produced the fixtures, so a breaking
    upstream release is identifiable from the failing CI run alone."""
    import importlib
    import json

    versions = {}
    for mod in ["xgboost", "lightgbm", "catboost", "sklearn", "skl2onnx", "onnx", "numpy"]:
        try:
            versions[mod] = importlib.import_module(mod).__version__
        except ImportError:
            pass
    out = base_dir / "versions.json"
    out.write_text(json.dumps(versions, indent=2) + "\n")
    print(f"Framework versions recorded in {out}: {versions}")


if __name__ == "__main__":
    main()
