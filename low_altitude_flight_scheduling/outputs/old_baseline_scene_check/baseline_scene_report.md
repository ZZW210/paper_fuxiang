# Historical Baseline Scene Check

No optimizer, Stage1, Stage2, FATA, seed scan, A* regeneration, OD sampling, or altitude reassignment was executed.

- Baseline source commit: `11e1a049d5f397362a97d7620ecaa4553209a6d4`
- Baseline plan SHA256: `5580bd2ad2cbdcea7dac0b3f8ac8dd990babce979aee4b274285986bc68facf7`
- Flight count: 100
- Current paper-strict conflict settings: `t_conflict=30.0`, `alpha=0.05`, `sigma0=1.0`, `sigma_rate=0.01`
- Route altitude distribution (grid z): `{0: 366, 1: 2611, 2: 1846, 3: 302}`
- Route altitude mean/min/max (grid z): `1.4066` / `0` / `3`
- Route points at z=0: `7.1415%`
- Non-endpoint route points at z=0: `3.3706%`
- Deterministic conflict points: 99
- Uncertain conflict points: 130
- Unique conflict pairs: 85
- Mean/max binary degree: `1.7000` / `5`
- CI l=2 Top10: `[41, 62, 18, 64, 83, 4, 16, 80, 49, 3]`
- Top10 conflict-point coverage: `0.246154`
- Top10 conflict-pair coverage: `0.341176`
