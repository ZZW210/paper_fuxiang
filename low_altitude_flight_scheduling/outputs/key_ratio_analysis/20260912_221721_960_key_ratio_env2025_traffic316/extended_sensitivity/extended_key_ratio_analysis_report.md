# Extended Key-Ratio Sensitivity Analysis

The 3%--12% rows are reused unchanged from the completed Table 9 reproduction debug experiment. Ratios above 0.12 are a **reproduction-specific extended sensitivity analysis**, not part of the paper's original Table 9 range. Every ratio uses the same fixed environment, traffic, initial CI ranking, FATA/ADM settings, and optimizer seeds 0, 1, and 2.

1. First ratio with a zero-conflict result: none in 3 repeats.
2. First ratio with all three runs at zero conflict: none in 3 repeats.
3. Best ratio under the stated ordering (lowest final mean conflicts, then fitness): 0.15 (K=15).
4. Best ratio is greater than the paper's 0.10.
5. Its initial CI conflict point coverage is 0.607477.
6. The fixed-scene result requires more key plans than 0.10, which is consistent with a different random network structure.
7. All larger tested ratios have higher mean fitness than the selected optimum, producing a local improve-then-worsen pattern.

No algorithm parameter was changed to make a ratio win. CI coverage is diagnostic only and is not the selection criterion.
