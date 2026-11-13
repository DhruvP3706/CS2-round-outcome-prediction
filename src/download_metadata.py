"""Download OpenCS2 metadata files from Hugging Face into data/opencs2/metadata/."""
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "opencs2" / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://huggingface.co/datasets/blanchon/opencs2_dataset/resolve/main"

FILES = {
    "matches.parquet": "index/matches.parquet",
    "rounds.parquet": "index/rounds.parquet",
    "kills.parquet": "events/kills.parquet",
    "round_player.parquet": "events/round_player.parquet",
    "pov_rounds.parquet": "index/pov_rounds.parquet",
    "smoke_flash_stats.parquet": "index/smoke_flash_stats.parquet",
    "duels.parquet": "events/duels.parquet",
}

def main():
    print(f"Target directory: {META_DIR}")
    for target_name, repo_rel in FILES.items():
        dst = META_DIR / target_name
        if dst.exists():
            print(f"[CACHED] {target_name} ({dst.stat().st_size / (1024*1024):.2f} MB)")
            continue
        url = f"{BASE_URL}/{repo_rel}"
        print(f"[DOWNLOADING] {url} -> {dst.name} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "wb") as f:
            f.write(resp.read())
        print(f"[OK] {target_name} ({dst.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
