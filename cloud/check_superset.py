"""Exit 0 if every results CSV in OLD_DIR has all its runs (by run_id) in the same-named CSV in NEW_DIR.

Used before overwriting results, so that a copy is never replaced by one that lacks
some of its rows:
    fetch.sh:    NEW = fetched copy, OLD = local copy   (local may be overwritten)
    teardown.sh: NEW = local copy,   OLD = fetched copy (instance may be destroyed)
Usage: python3 cloud/check_superset.py NEW_DIR OLD_DIR
"""

import csv
import sys
from pathlib import Path


def run_ids(path: Path) -> set[str]:
    with path.open(newline="") as f:
        return {row["run_id"] for row in csv.DictReader(f) if row.get("run_id")}


def main() -> None:
    new_dir, old_dir = Path(sys.argv[1]), Path(sys.argv[2])
    ok = True
    for old in sorted(old_dir.glob("*.csv")):
        new = new_dir / old.name
        old_ids = run_ids(old)
        new_ids = run_ids(new) if new.exists() else set()
        missing = old_ids - new_ids
        print(f"  {old.name}: {len(old_ids)} runs in {old_dir}, {len(new_ids)} in {new_dir}"
              + (f", {len(missing)} MISSING from the latter" if missing else ""))
        ok &= not missing
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
