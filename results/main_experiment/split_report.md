# Grouped split

       series  maps  rounds    ct     t
split                                  
test       54     7    2503  1260  1243
train     251     7   11576  5939  5637
val        54     7    2448  1236  1212

Series = match_id; maps nested within series. Seed 42. All models retain the metadata eligibility cohort; unavailable snapshots are imputed using training data only.

split  map_name   
test   de_ancient      449
       de_anubis       236
       de_dust2        476
       de_inferno      321
       de_mirage       456
       de_nuke         388
       de_overpass     177
train  de_ancient     2082
       de_anubis       959
       de_dust2       2235
       de_inferno     1311
       de_mirage      2125
       de_nuke        1782
       de_overpass    1082
val    de_ancient      335
       de_anubis       185
       de_dust2        616
       de_inferno      245
       de_mirage       473
       de_nuke         436
       de_overpass     158

Eligible rounds by cutoff:

       eligible_5s  eligible_10s  eligible_15s  eligible_20s  eligible_25s  eligible_30s  eligible_35s  eligible_40s  eligible_45s  eligible_50s  eligible_60s  eligible_75s
split                                                                                                                                                                       
test          2503          2503          2500          2489          2470          2427          2378          2325          2247          2155          1962          1566
train        11576         11576         11572         11529         11406         11197         10914         10552         10167          9740          8813          6983
val           2448          2448          2447          2435          2415          2372          2313          2248          2162          2086          1846          1470