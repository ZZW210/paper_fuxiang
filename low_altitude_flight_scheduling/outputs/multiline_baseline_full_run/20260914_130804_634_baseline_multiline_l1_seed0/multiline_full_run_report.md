# Multi-degree CI l=1 Full Baseline Run

1. Frozen baseline loaded: yes. Source commit `11e1a049d5f397362a97d7620ecaa4553209a6d4`.
2. Baseline hash validated before and after the run: `5580bd2ad2cbdcea7dac0b3f8ac8dd990babce979aee4b274285986bc68facf7`.
3. Initial deterministic/uncertain conflict points: 99 / 130; unique pairs: 85.
4. Multi CI l=1 Top10: `[52, 77, 87, 70, 41, 62, 16, 83, 4, 18]`.
5. Dense-flight results: 52, 77, 70, and 87 are all in the Multi Top10, with CI_multi_l1 420, 420, 196, and 210 respectively. Their full Stage1/Stage2 changes are in `dense_flights_52_77_70_87.csv`.
6. Multi Top10 covers 61 / 130 conflict points (46.9231%); unique-pair coverage is 29.4118%.
7. Stage1: 130 -> 69 (46.9231% reduction).
8. Stage2: 69 -> 9 (86.9565% reduction).
9. Zero conflict: False.
10. Final provenance: exact survivors=4, moved same pair=0, new-pair points=5.
11. Risk change: -0.1867%; delayed flights=28; changed flights=50.
12. Runtime: Stage1=20.759s; Stage2=45.321s; total=66.936s.
13. The optimizer completed before this report was written. The report-generation repair did not rerun or change Stage1, Stage2, FATA, the frozen baseline, or their outputs.
