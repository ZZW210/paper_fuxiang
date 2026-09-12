# Implementation assumptions


## Population and risk implementation assumptions

The target paper specifies 3500 persons/km^2 and a simulated building map.
Reference [28] supplies the population-center/gravity concept, not Singapore
MRT data, station locations, trained predictors or its measured densities.
Its repository PDF returned HTTP 405 during this implementation; the exact
piecewise diffusion equation follows the user-supplied specification, rather
than a claim that the full reference dataset/implementation was reproduced.

Implementation assumptions: non-overlapping 10x10-cell windows; window
centroids as synthetic centers; eight-neighbor local maxima including tied
plateaus; descending footprint density then x/y tie breaking; Euclidean NMS
with minimum spacing 1000m; at most four centers. Fewer admissible maxima are
recorded, never replaced or selected against conflict statistics. Buildings
are counted once in their horizontal footprint. Normalized building density
is window density divided by the maximum density over all windows (zero for
an empty city). Center strength is 3500*(1+beta*normalized_density).
Four centers and beta=2 are not disclosed paper parameters. Explicit future
beta sensitivity values are 1,2,4; there is no automatic parameter selection.

Only the nearest center contributes: sigma_center*exp(1-r_km^2) for r_km<1;
otherwise 3500. Boundary discontinuity is preserved, with no smoothing or
center summation. Building cells retain population; obstacles remain separate.
Population arrays/CSV use persons/km^2; risk impact areas use m^2 after exactly
one division by 1,000,000. Reference [28] is time-dependent, but the target paper
does not disclose a time-of-day population scenario; therefore a static spatial
snapshot using its gravity diffusion concept is used.

Existing ballistic/parachute severity and drift approximations are unchanged
in this population-only comparison, retaining P_total=P_bal+P_par and
P_r=P_failure*P_impact*P_severity, P_impact=A*rho. They are not a claim of an
exact reconstruction of every descent equation. risk_map_raw.npy stores these
unnormalized probabilities; risk_map.npy keeps the existing global min-max
A* normalization, an implementation assumption shared by all four experiments.
Strict risk values are not overwritten by obstacle occupancy. The old_gaussian
experiment retains the original population generator (including footprint
boost) but uses the same risk computation as the gravity experiments.

z1..z4 name the existing 15/45/75/105m cell-center layers (one-based).
risk_map_ground.png is the lowest-layer ground-risk projection, not a fifth
planning layer or a zero-altitude crash simulation. Risk statistics describe
the normalized map; raw risk is separately available for formula/unit auditing.
