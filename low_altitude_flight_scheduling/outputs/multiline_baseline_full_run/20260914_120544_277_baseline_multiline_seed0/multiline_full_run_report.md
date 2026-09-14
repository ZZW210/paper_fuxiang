# Multi-degree CI Full Baseline Run

1. Frozen baseline loaded: yes. Source commit `11e1a049d5f397362a97d7620ecaa4553209a6d4`.
2. Baseline hash validated before and after the run: `5580bd2ad2cbdcea7dac0b3f8ac8dd990babce979aee4b274285986bc68facf7`.
3. Initial deterministic/uncertain conflict points: 99 / 130; unique pairs: 85.
4. Multi CI l=2 Top10: `[41, 62, 64, 4, 83, 18, 16, 56, 78, 80]`.
5. Dense-flight results:
- Flight 52: multi degree=21, CI_multi_l2=0.0, rank=75, Multi Top10=False.
- Flight 77: multi degree=22, CI_multi_l2=0.0, rank=90, Multi Top10=False.
- Flight 70: multi degree=15, CI_multi_l2=14.0, rank=14, Multi Top10=False.
- Flight 87: multi degree=15, CI_multi_l2=0.0, rank=94, Multi Top10=False.
6. Multi Top10 covers 32 / 130 conflict points (24.6154%); unique-pair coverage is 31.7647%.
7. Stage1: 130 -> 98 (24.6154% reduction).
8. Stage2: 98 -> 11 (88.7755% reduction).
9. Zero conflict: False.
10. Final provenance: exact survivors=9, moved same pair=0, new-pair points=2.
11. Risk increase: 0.1634%; delayed flights=27; changed flights=52.
12. Runtime: Stage1=19.698s; Stage2=49.414s; total=69.948s.
13. Multi-degree correctly exposes the repeated-conflict incident count, but this fixed l=2 CI did not select the dense flights in this baseline because their simple-topology two-hop shell contribution is zero or small. Multi-degree CI explicitly uses full incident-record degree; this run reports the actual scheduling outcome and does not alter the paper-strict optimizer to favor that conclusion.
