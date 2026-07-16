import os
import subprocess
import sys
from pathlib import Path

def main():
    base_dir = Path("assets/tests")
    if not base_dir.exists():
        print(f"Error: {base_dir} not found. Run from project root.")
        sys.exit(1)

    scripts = list(base_dir.glob("**/generate.py"))
    scripts.sort()

    if not scripts:
        print("No generate.py scripts found.")
        return

    print(f"Found {len(scripts)} generation scripts.")

    for script in scripts:
        print("-" * 60)
        print(f"Running: {script}")
        
        try:
            # Run the script in its own directory
            subprocess.run(
                [sys.executable, "generate.py"],
                cwd=script.parent,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
            # We continue with other scripts even if one fails
            # or we could exit(1) if we want it to be strict
        except Exception as e:
            print(f"Unexpected error: {e}")

    print("-" * 60)
    print("Finished processing all generation scripts.")

if __name__ == "__main__":
    main()
