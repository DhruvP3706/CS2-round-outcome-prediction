# Step 5 summary (cumulative ablation, fixed LightGBM, full train, val-only thresholds)
A = Step2 meta + Step4 player state (97 feats) ... H = A + 7 families (202 feats). Step 4 results untouched.
Best test macro F1 (any set): 20s .678 (C) | 30s .701 (G) | 40s .727 (A) | 50s .761 (F) | 60s .787 (C) | 75s .793 (A). Step 4 best: .680 .702 .727 .760 .780 .796.
Mean delta vs A for t<=75: B -.002 C +.001 D -.001 E -.001 F -.001 G -.002 H -.001. No family gives a reliable gain.
Seed-noise (A, seeds 1/2): +-.003-.007 at 20-75s, i.e. as large as any observed delta.
Only 2 of 84 (set,t<=75) paired-vs-previous CIs exclude 0 positively (C @20s +.011 [.004,.020]) and 1 negative (D @20s -.010): chance level.
Drop all families. Bomb site not used (rounds.bomb_site is unreliable: only 'bombsite_b'/'not_planted').
Leakage test (T=10,30,60,90): all features unchanged after scrambling post-cutoff kills, plants and snapshots.
