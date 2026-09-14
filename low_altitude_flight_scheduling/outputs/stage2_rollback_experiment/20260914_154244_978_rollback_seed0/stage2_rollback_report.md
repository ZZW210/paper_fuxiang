# Stage2 Global Rollback Diagnostic

This is a diagnostic experiment, NOT a claim that the paper contains rollback.
Branch: scheduling. Frozen source snapshot: `ebc5b9cfbf689eda0ee0b1baf7439def510d2cdf`. Baseline hash: `5580bd2ad2cbdcea7dac0b3f8ac8dd990babce979aee4b274285986bc68facf7`.
Only Stage2 candidate local-action acceptance changed. No repair, new candidates, seeds, or parameter tuning.
ADM and FATA ran for 200 generations with NP=50, Parf=0.2. Optimizer base seed=0;
Stage2 RNG seed=206 is the unchanged existing scheduler offset, not an extra seed search.

| Metric | NO ROLLBACK | WITH ROLLBACK |
| --- | --- | --- |
| final_conflict_points | 9 | 2 |
| final_conflict_pairs | 9 | 2 |
| final_exact_survivors | 4 | 2 |
| final_moved_same_pair | 0 | 0 |
| final_new_conflict_points | 5 | 0 |
| risk_increase_percent | -0.18665761357711955 | -1.523310030610403 |
| delayed_flights | 28 | 29 |
| changed_flights | 50 | 44 |
| stage2_runtime_sec | 45.32148839999991 | 1393.169037900001 |

1. Stage1 strictly reproduced 130 -> 69, 60 pairs, identical fitness 2927289.6305803587; Top10 `[52, 77, 87, 70, 41, 62, 16, 83, 4, 18]`. No Stage1 modifications.
2. Final: 69 -> 2 points, 2 pairs.
3. Zero conflict: False.
4. New-pair points: 5 -> 0. Original new flight pairs still present: []. This checks pairs, not merely aggregate counts.
5. Search rollback count=208773; accepted actions=179498. Pair increase rejects=50441; equal pairs with non-decreased points rejects=158332.
6. Exact survivors: 4 -> 2; moved same pair: 0 -> 0; new points: 5 -> 0. Old-conflict increase with fewer new conflicts: False.
7. Stage2 time increased by 1347.848s (30.740x); baseline Stage1=20.759s, rerun Stage1=20.080s. Timing includes validation/logging overhead and machine variability.
8. Risk change=-1.523310%; worse than no rollback: False.
9. Delayed flights: 28 -> 29; increased: True. Changed flights: 50 -> 44.
10. This single-scene/single-seed diagnostic cannot establish lack of global protection as the main cause. It demonstrates the observed trade-off above. A strict lexicographic improvement can still create a new pair if more old pairs disappear; suppression of every new pair is NOT guaranteed.

## Action Semantics And Coordination Limitation

Each candidate starts from the same Stage1 schedule. Actions follow unchanged block order and schedule/speed/reroute order.
Existing shared per-flight ATD and selected speed segments / merged route windows are applied together per flight x strategy,
with all incident conflict indices logged. No duplicate conflicting assignments or extra decision variables were introduced.
All 100 flights are fully detected before and after EACH action; equal conflict counts are rejected even if fitness improves.
Infeasible route candidates retain the original infinite-fitness rule and are not converted into repair candidates.
Rollback restores the exact prior actor reference; other flight plans are immutable during that action.
Rejecting one action can prevent a later cooperative combination from improving. The experiment deliberately retains this side effect.
ADM still learns sampled species without filtering rejected strategy labels; no acceptance-adaptive probability update was added.

Search counts include worker candidate evaluations, incumbent reevaluations, and ADM's final decode, including attempted actions
in candidates that later become geometrically infeasible. They are NOT unique actions committed to the final schedule.
Final selected-plan replay counts: `{'rollback_count': 20, 'accepted_action_count': 35, 'pairs_increased_count': 2, 'same_pairs_points_not_decreased_count': 18}`; detailed trace: `stage2_selected_action_log.csv`.
Worker logs are retained in `search_actions/`; aggregate full trace is `stage2_action_log.csv`; rejected actions are `stage2_rollback_log.csv`.
The final replay is outside measured optimizer runtime and agrees exactly with ADM's returned fitness.

## Interpretation And Verification

In this fixed scene/seed, protection reduces final conflicts from 9 to 2, removes all five original new pairs,
and reduces exact survivors from 4 to 2. There is no "more old conflicts for fewer new conflicts" outcome here.
This supports global validation as a useful protection against newly introduced conflicts in this instance.
It does NOT establish it as the sole or main reason strict Stage2 cannot reach zero: two original conflicts remain,
and per-action rejection can block coordinated changes. No separate coordination ablation or additional seed was run.
The cost is 1347.848 additional seconds and one additional delayed flight, despite lower risk and fewer changed flights.

All 388271 search actions and 55 selected-plan actions pass the exact acceptance and restored-state chain audit.
The two residual pairs are (7, 27) and (36, 94). Full global conflict records and provenance are independently
rechecked from the serialized final plans; details are recorded in `action_audit.json`.
Regression verification: 146 tests passed. The initial attempt stopped at generation 1 due solely to the
progress callback returning None; it was marked failed and the complete run used the identical seed/configuration.
