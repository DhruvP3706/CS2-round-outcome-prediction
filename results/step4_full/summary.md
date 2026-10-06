# Step 4 full-data validation (full train: 555 series-maps; val 119; test 120 map files; 794 files, 18.09 GB streamed)
Models: A meta (63 feats) | B +player state (97) | C +spatial (306) | D +movement (314). LightGBM, no tuning, val-only threshold, test once.
Mean dF1 (t<=60 / all 15 times): B-A +0.0079/+0.0068 | C-B +0.0026/+0.0019 | D-B +0.0024/+0.0021 | D-C -0.0002/+0.0015 | A - Step2 +0.0003/+0.0020.
Only B-A at 20s & 25s, C-B at 5s, D-C at 20s have 95% series-bootstrap CIs excluding 0 (15 times x 4 comparisons => consistent with chance for the latter two).
Best F1 <60s: 0.760 (C and D at 50s). 60s: 0.780 (B, C). 75s: 0.796 (D).
Conclusion: player state gives a small, fairly consistent gain; spatial and movement add ~nothing measurable. 0.80 not reached before 75s.
Files: results.csv (all models/times), comparison.csv (paired deltas + CIs), ../../plots/step4_full_f1_vs_time.png. Pilot results in results/step4 untouched.
