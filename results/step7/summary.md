# Step 7 summary: sub-second dynamics pilot (A=97 baseline, B player-state changes 30, C movement 42, D spatial 22, E engagement 14, F all = 205)
Config fixed (Step 6 c02: leaves 4, depth 3, min_child 80), seed 42; seeds 1,2 for noise. 794 tick files re-streamed (column projection, nothing kept) for 30/40/50/60/75s windows (0.5/1/2 s, backward only).
Test dF1 vs A (seed 42), range over the 5 times: B -.001..+.003, C -.006..+.008, D -.002..+.006, E -.002..+.002, F -.005..+.005. 3-seed mean of means: B +.0002 C -.0013 D +.0006 E -.0007 F -.0010.
Validation dF1 vs A: all within +-.006 (only D@30s CI excludes 0, negative). No family reaches +0.01 at any time; no bootstrap CI excludes 0 on test.
Seed spread of A alone: .003-.005. Conclusion: short-term dynamics in this representation are not useful. Abandon.
Leakage: scrambling all tick rows and kill events after T=30/50/60 leaves all features at t<=T unchanged; windows end exactly at pred tick.
