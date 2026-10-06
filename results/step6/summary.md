# Step 6 summary
Features: 97 (Step 2 meta + Step 4 player state). 13 configs searched at 30/40/50/60/75s on train+val only; selection = max mean val macro F1 (seed 42).
Baseline reproduction: exact (max |diff| 0.0000 vs Step 4 model B, all 15 times).
Selected: c02 num_leaves=4, max_depth=3, min_child_samples=80 (others as baseline). Mean val macro F1 0.7812 vs baseline 0.7776 (+0.0036).
Test (seed 42) delta vs baseline: 30s +.012, 40s +.001, 50s +.001, 60s .000, 75s +.002; mean over 15 times ~ +.001. Seed-averaged (3 seeds) delta at focus: +.007,+.004,+.005,+.001,+.002.
Baseline seed range (3 seeds) at focus is .003-.008 wide, so the gain is at/below the noise floor except possibly 30s.
Conclusion: configuration is not the bottleneck; simpler trees are marginally better. Stop tuning.
