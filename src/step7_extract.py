"""Step 7a: stream map-level tick files (existing 794 files, column projection, no raw ticks kept) and keep only
window-derived dynamics rows for t in 30/40/50/60/75 s. Resumable: data/features/step7/tickdyn/{match}_{map}.parquet"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402

from cs2pred.config import PROCESSED_DIR  # noqa: E402
from cs2pred.step7_features import RAW_COLS, extract_dynamics  # noqa: E402
from step3_tick_probe import WDS, list_tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "features" / "step7" / "tickdyn"
FOCUS = [30, 40, 50, 60, 75]
TABLE = None


def read_ticks(path):
    for attempt in range(4):
        try:
            with HfFileSystem().open(path, "rb") as fh:
                return pq.ParquetFile(fh).read(columns=RAW_COLS).to_pandas()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def process(job):
    mid, mp, path = job
    fn = OUT / f"{mid}_{mp}.parquet"
    if fn.exists():
        return mid, mp, "cached", 0
    rt = TABLE[(TABLE.match_id == mid) & (TABLE.map_name == mp)].set_index("round")
    try:
        d = extract_dynamics(read_ticks(path), rt, FOCUS)
    except Exception as e:
        return mid, mp, f"FAILED {e!r}", 0
    d.insert(0, "map_name", mp); d.insert(0, "match_id", mid)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(fn.with_suffix(".tmp")); fn.with_suffix(".tmp").rename(fn)
    return mid, mp, "ok", len(d)


def _init(t):
    global TABLE
    TABLE = t


def jobs():
    fs = pd.DataFrame(list_tree(WDS, "ticks"))
    fs["match_id"] = fs.path.str.extract(r"match_id=(\d+)")[0].astype(int)
    fs["map_name"] = fs.path.str.extract(r"map_name=(de_\w+)/")[0]
    return [(r.match_id, r.map_name, f"datasets/{WDS}/{r.path}") for r in fs.itertuples()], fs["size"].sum()


if __name__ == "__main__":
    table = pd.read_parquet(PROCESSED_DIR / "round_table.parquet")
    js, sz = jobs()
    if len(sys.argv) > 1:
        js = js[:int(sys.argv[1])]
    print(f"{len(js)} files, {sz / 1e9:.2f} GB (streamed, not stored)", flush=True)
    t0, bad = time.time(), []
    with Pool(8, initializer=_init, initargs=(table,)) as pool:
        for i, (mid, mp, st, n) in enumerate(pool.imap_unordered(process, js), 1):
            if st.startswith("FAILED"):
                bad.append((mid, mp, st)); print(mid, mp, st, flush=True)
            if i % 20 == 0 or i == len(js):
                print(f"[{i}/{len(js)}] rows={n} {time.time() - t0:.0f}s", flush=True)
    print("DONE failed", bad, flush=True)
