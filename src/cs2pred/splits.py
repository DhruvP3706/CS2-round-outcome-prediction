"""Series-level train/validation/test split (all maps of a series stay together)."""
import numpy as np
import pandas as pd

SPLIT_SEED = 42


def series_split(table: pd.DataFrame, fracs=(0.70, 0.15, 0.15), seed: int = SPLIT_SEED) -> pd.Series:
    """Return split label ('train'/'val'/'test') per match_id (series)."""
    ids = np.sort(table["match_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr, n_va = int(round(fracs[0] * n)), int(round(fracs[1] * n))
    lab = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va))
    return pd.Series(lab, index=ids, name="split")
