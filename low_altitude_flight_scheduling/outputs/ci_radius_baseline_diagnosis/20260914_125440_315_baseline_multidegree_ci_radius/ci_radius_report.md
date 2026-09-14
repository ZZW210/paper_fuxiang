# Multi-degree CI Radius Diagnosis

Frozen baseline hash: `5580bd2ad2cbdcea7dac0b3f8ac8dd990babce979aee4b274285986bc68facf7`. No generator, optimizer, Stage1, Stage2, or FATA was run.

- Initial deterministic / uncertain conflicts / unique pairs: 99 / 130 / 85.
- Top10 l=1: `[52, 77, 87, 70, 41, 62, 16, 83, 4, 18]`
- Top10 l=2: `[41, 62, 64, 4, 83, 18, 16, 56, 78, 80]`
- Top10 l=3: `[83, 62, 16, 64, 4, 78, 18, 56, 41, 53]`
- Top10 l=4: `[83, 56, 89, 78, 53, 2, 16, 64, 31, 97]`

## Dense flights

The full manual shell calculation is in `ci_radius_dense_audit.md`; compact values are in `ci_radius_dense_flights.csv`.

- Flight 52: l=1 CI=420.0 rank=1, l=2 CI=0.0 rank=75, l=3 CI=0.0 rank=69, l=4 CI=0.0 rank=68
- Flight 77: l=1 CI=420.0 rank=2, l=2 CI=0.0 rank=90, l=3 CI=0.0 rank=86, l=4 CI=0.0 rank=85
- Flight 70: l=1 CI=196.0 rank=4, l=2 CI=14.0 rank=14, l=3 CI=0.0 rank=79, l=4 CI=0.0 rank=78
- Flight 87: l=1 CI=210.0 rank=3, l=2 CI=0.0 rank=94, l=3 CI=0.0 rank=92, l=4 CI=0.0 rank=91

## Comparative evidence

- Highest Top10 conflict-point coverage: l=[1]; this is a record-level count with every covered conflict counted once.
- Highest mean Multi-degree among Top10: l=[1].
- Highest mean unique conflict partners among Top10: l=[2].
- Most repeated endpoint selection: l=[1]; its redundancy ratio is reported in `ci_radius_top10_summary.csv` and is not a coverage gain.
- Strongest Top10 removal by final LCC size: l=[1, 2, 3, 4]; the attack diagnostic does not distinguish these radii when tied.

l=2 can give a high-Multi-degree node CI=0 because the formula requires a nonzero sum over the exact simple-topology two-hop shell. Repeated parallel conflicts increase D_multi but do not create additional shell nodes. Here, flight 52 has only flight 76 in its l=2 shell and D_multi(76)-1=0; the full arithmetic is in the dense audit.

Conclusion: l=1 most directly reflects the repeated-conflict dense pairs in this frozen baseline and has the highest Top10 point coverage, while l=2 has the highest Top10 partner diversity and lower redundancy. The metrics conflict, so this baseline cannot uniquely choose l or establish the paper's undisclosed radius.
