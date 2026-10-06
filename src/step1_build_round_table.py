"""Step 1: build the canonical round table, run validation, write eligibility summary."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

from cs2pred.config import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from cs2pred.round_table import build_round_table, eligibility_summary  # noqa: E402
from cs2pred.validation import validate  # noqa: E402


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = build_round_table()
    table.to_parquet(PROCESSED_DIR / "round_table.parquet", index=False)

    val = validate(table)
    val.to_csv(RESULTS_DIR / "validation_report.csv", index=False)
    summ = eligibility_summary(table)
    summ.to_csv(RESULTS_DIR / "eligibility_summary.csv", index=False)
    (RESULTS_DIR / "class_balance_by_map.csv").write_text(
        table.groupby("map_name")["y_ct_win"].agg(["size", "mean"]).to_csv())

    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(f"round table: {table.shape}, maps={table[['match_id','map_name']].drop_duplicates().shape[0]}, "
          f"series={table.match_id.nunique()}")
    print("\nVALIDATION"); print(val.to_string(index=False))
    print("\nELIGIBILITY"); print(summ.round(4).to_string(index=False))
    if (val.status == "FAIL").any():
        sys.exit(1)


if __name__ == "__main__":
    main()
