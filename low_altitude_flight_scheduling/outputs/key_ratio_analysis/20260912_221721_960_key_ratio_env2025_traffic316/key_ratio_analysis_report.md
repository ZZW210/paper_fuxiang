# Key Flight Ratio Analysis

Each ratio was independently optimized 3 times to reduce the influence of randomized optimization. This does not assert that the paper's Table 9 used 30 repetitions.

## Answers

A. The best key-flight ratio in this fixed scene is 0.10 (K=10), selected by lowest mean remaining conflicts followed by lowest mean final fitness.

B. It is consistent with the paper's 0.10.

C. Difference from 0.10: +0.00.

D. Initial CI conflict point coverage is monotonically non-decreasing because each K is a longer prefix of one fixed CI ranking.

E. Higher coverage need not keep improving final optimization: it also enlarges the Stage1 joint decision space and can increase FATA search difficulty, while Stage2 independently assigns strategies to residual conflict points.

F. The result should be interpreted by the small-ratio capacity versus larger-ratio decision-dimension trade-off shown in the measured table, rather than forcing 0.10 to win.

G. The comparison figure preserves raw values. It can compare trends with Table 9, but numerical agreement is not expected because the paper's original random scene and seeds are unavailable.

The paper's 0.10 was obtained in its specific simulation scenario. This reproduction uses a fixed synthetic environment and traffic seed, so a different optimum is allowed.
