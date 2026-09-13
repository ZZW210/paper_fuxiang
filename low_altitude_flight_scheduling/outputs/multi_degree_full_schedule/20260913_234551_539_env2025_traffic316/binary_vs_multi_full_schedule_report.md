# Binary vs Multi-degree Full Schedule

Binary Top10: [98, 47, 63, 49, 56, 25, 72, 64, 1, 4]
Multi Top10: [73, 47, 49, 56, 33, 78, 42, 25, 1, 63]
Overlap: 6.

## Summary
- method=binary, n_runs=3, key_conflict_point_coverage=0.5233644859813084, stage1_conflicts_mean=52.333333333333336, stage1_conflicts_std=1.5275252316519468, stage1_reduction_mean=0.5109034267912773, final_conflicts_mean=7.333333333333333, final_conflicts_std=1.5275252316519465, zero_conflict_success_rate=0.0, changed_flights_mean=44.666666666666664, delayed_flights_mean=23.0, risk_increase_mean=0.5596128810067689, risk_increase_std=0.5793789794274181, final_fitness_mean=315419.1784375419, stage1_runtime_mean=20.01664783332914, stage2_runtime_mean=37.42453113333128, total_runtime_mean=57.47434253333389
- method=multi, n_runs=3, key_conflict_point_coverage=0.48598130841121495, stage1_conflicts_mean=55.333333333333336, stage1_conflicts_std=0.5773502691896263, stage1_reduction_mean=0.48286604361370716, final_conflicts_mean=11.666666666666666, final_conflicts_std=2.5166114784235836, zero_conflict_success_rate=0.0, changed_flights_mean=46.666666666666664, delayed_flights_mean=22.0, risk_increase_mean=0.3249872624460192, risk_increase_std=0.5662391193889899, final_fitness_mean=497152.0431784133, stage1_runtime_mean=19.848021899997548, stage2_runtime_mean=35.66651789999984, total_runtime_mean=55.55143280000144

Delta_stage1=-3.000. Conclusion: Stage2 remains the main bottleneck.
Risk and changed/delayed-flight metrics are in the summary; no parameter other than key-flight ranking changed.
