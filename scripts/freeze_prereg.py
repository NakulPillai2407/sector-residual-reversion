"""Write results/FREEZE.json (SHA256 of config.yaml, membership panel, data manifest), commit and tag `prereg-frozen`."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CONFIG_PATH, PROC_DIR, RESULTS_DIR, ROOT, sha256_file  # noqa: E402


def freeze_hashes() -> dict[str, str]:
    return {"config.yaml": sha256_file(CONFIG_PATH),
            "data/processed/membership.parquet": sha256_file(PROC_DIR / "membership.parquet"),
            "data/MANIFEST.json": sha256_file(ROOT / "data" / "MANIFEST.json")}


def main() -> None:
    man = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
    bad = [k for k, h in man.items() if sha256_file(ROOT / k) != h]
    if bad:
        raise SystemExit(f"MANIFEST does not match files on disk: {bad}")
    (RESULTS_DIR / "FREEZE.json").write_text(json.dumps(freeze_hashes(), indent=1))
    git = lambda *a: subprocess.run(["git", "-c", "user.name=Nakul Pillai", "-c", "user.email=pillainakul@gmail.com", *a], cwd=ROOT, check=True, capture_output=True, text=True)  # noqa: E731
    git("add", "-A")
    git("commit", "-m", "Freeze pre-registration before Stage 3 holdout\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>")
    git("tag", "prereg-frozen")
    print(json.dumps(freeze_hashes(), indent=1), "\ntag prereg-frozen ->", git("rev-parse", "HEAD").stdout.strip())


if __name__ == "__main__":
    main()
