# Decentralized checkpoint evaluation report

This report compares the centralized-trained checkpoint deployed with
*decentralized* per-agent observation against the fully-active ACS
baseline on a range of communication topologies.

**Convergence criterion.** An episode is *converged* if there exists a trailing window of `100` steps in which (a) connectivity held (single component every step), (b) `std_pos` varied by less than `2.0` m, and (c) the polar order parameter varied by less than `0.05`.

## Network topology descriptions

| topology | description |
|---|---|
| `fully_connected` | Every agent connects to every other. N(N-1)/2 edges. Baseline topology used for training checkpoint 74. |
| `star` | Hub (index 0) connects to all others; leaves connect only to hub. 19 edges for 20 agents. |
| `wheel` | Hub connects to all outer agents + outer agents form a ring. Each outer agent sees hub + 2 ring neighbors = 3 connections. |
| `binary_tree` | Heap-style binary tree: parent of i (i>=1) is (i-1)//2. Root sees 2 children; internal nodes see parent + 2 children; leaves see parent only. |
| `line` | Path graph 0-1-2-...-19. Ends see 1 neighbor; middle agents see 2. Ring without wraparound. |
| `ring_k1` | Circular ring k=1: each agent connects to 1 neighbor on each side. 2 connections per agent. |
| `ring_k2` | Ring k=2: each agent connects to 2 neighbors on each side. 4 connections per agent. |
| `ring_k3` | Ring k=3: each agent connects to 3 neighbors on each side. 6 connections per agent. |
| `mst` | Minimum spanning tree of pairwise distances. Dynamic (rebuilt each step). Always connected, N-1 edges, avg degree ~2. |
| `delaunay` | Delaunay triangulation of positions. Dynamic. Always connected, typically ~3N edges, avg degree ~6. |
| `knn_k5` | k-nearest neighbors (k=5), symmetrized. Dynamic. May disconnect if clusters separate. |
| `knn_k10` | k-nearest neighbors (k=10), symmetrized. Dynamic. Dense enough to stay connected at convergence. |
| `knn_k15` | k-nearest neighbors (k=15), symmetrized. Dynamic. Nearly FC (15/19 possible neighbors). |
| `disk_R60` | Agents within 60m connected. Dynamic. Too sparse — swarm disperses. |
| `disk_R120` | Agents within 120m connected. Dynamic. Effectively FC at ACS convergence but policy-induced spread breaks it. |
| `er_p0.2` | Erdos-Renyi, edge prob 0.2. Static per episode. Expected degree 3.8. Connected ~70%. |
| `er_p0.3` | Erdos-Renyi, edge prob 0.3. Static per episode. Expected degree 5.7. Connected ~97%. |
| `er_p0.5` | Erdos-Renyi, edge prob 0.5. Static per episode. Expected degree 9.5. Connected ~100%. |

## Metric definitions

- **R**: mean cumulative episode reward (higher = better).
- **d-a**: reward gap between decentralized policy and ACS (positive = policy beats ACS).
- **cv%**: fraction of episodes that meet the convergence criterion.
- **p50 / p90** (conv step): median / 90-th percentile of the first-converged step across converged episodes.
- **1f%**: fraction of episodes whose final step is a single network component ("single flock").
- **conn%**: mean fraction of steps per episode in which the network was a single component.
- **op**: polar order parameter at last step `|mean(v_i)| / speed` (0 = random, 1 = aligned).
- **pos**: `sqrt(Var(x) + Var(y))` at last step — spatial tightness.

## Convergence budget

| topology | dec cv% | acs cv% | dec p50 | dec p90 | dec max | acs p50 | acs p90 | acs max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fully_connected` | 100% | 100% | 331 | 471 | 694 | 395 | 721 | 1021 |
| `star` | 43% | 100% | 1179 | 2171 | 2478 | 574 | 1300 | 2023 |
| `wheel` | 63% | 77% | 1214 | 2045 | 2480 | 1153 | 1909 | 2473 |
| `binary_tree` | 16% | 51% | 1852 | 2303 | 2451 | 1485 | 2182 | 2474 |
| `line` | 16% | 48% | 1730 | 2209 | 2350 | 1784 | 2346 | 2498 |
| `ring_k1` | 9% | 25% | 1556 | 1979 | 2403 | 1572 | 2110 | 2326 |
| `ring_k2` | 35% | 51% | 1495 | 2201 | 2387 | 1476 | 2055 | 2470 |
| `ring_k3` | 81% | 69% | 1223 | 1952 | 2303 | 1293 | 2197 | 2468 |
| `mst` | 70% | 94% | 1424 | 2262 | 2494 | 1341 | 1932 | 2460 |
| `delaunay` | 42% | 57% | 1581 | 2366 | 2496 | 1475 | 2213 | 2479 |
| `knn_k5` | 34% | 37% | 820 | 1351 | 1874 | 608 | 795 | 909 |
| `knn_k10` | 100% | 100% | 368 | 572 | 770 | 383 | 615 | 1206 |
| `knn_k15` | 100% | 100% | 312 | 446 | 647 | 365 | 612 | 1031 |
| `disk_R60` | 0% | 0% | - | - | - | - | - | - |
| `disk_R120` | 58% | 83% | 425 | 599 | 747 | 417 | 649 | 1019 |
| `er_p0.2` | 34% | 41% | 1407 | 2420 | 2498 | 1210 | 2172 | 2471 |
| `er_p0.3` | 84% | 79% | 1024 | 2089 | 2486 | 1148 | 2185 | 2435 |
| `er_p0.5` | 100% | 95% | 613 | 1278 | 2378 | 761 | 1569 | 2261 |

## Overall results (all episodes)

| topology | dec R | acs R | d-a | dec 1f% | acs 1f% | dec conn% | acs conn% | dec op | acs op | dec pos | acs pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fully_connected` | -351.7 | -425.3 | +74 | 100% | 100% | 100% | 100% | 1.000 | 1.000 | 39.5 | 39.6 |
| `star` | -1060.3 | -530.5 | -530 | 100% | 100% | 100% | 100% | 0.642 | 1.000 | 67.9 | 47.0 |
| `wheel` | -1018.0 | -1014.2 | -4 | 100% | 100% | 100% | 100% | 0.808 | 0.868 | 61.0 | 53.6 |
| `binary_tree` | -1168.0 | -1030.8 | -137 | 100% | 100% | 100% | 100% | 0.468 | 0.736 | 133.1 | 108.8 |
| `line` | -1123.7 | -1021.8 | -102 | 100% | 100% | 100% | 100% | 0.498 | 0.746 | 202.2 | 192.6 |
| `ring_k1` | -1230.8 | -1211.3 | -19 | 100% | 100% | 100% | 100% | 0.393 | 0.579 | 149.0 | 136.6 |
| `ring_k2` | -1145.5 | -1153.2 | +8 | 100% | 100% | 100% | 100% | 0.653 | 0.731 | 85.3 | 73.7 |
| `ring_k3` | -821.3 | -1049.6 | +228 | 100% | 100% | 100% | 100% | 0.894 | 0.847 | 56.3 | 55.4 |
| `mst` | -726.7 | -611.6 | -115 | 100% | 100% | 100% | 100% | 0.894 | 0.988 | 228.8 | 216.9 |
| `delaunay` | -1115.4 | -1100.8 | -15 | 100% | 100% | 100% | 100% | 0.805 | 0.856 | 98.7 | 93.5 |
| `knn_k5` | -430.4 | -419.1 | -11 | 34% | 36% | 39% | 41% | 0.786 | 0.787 | 1459.0 | 1413.5 |
| `knn_k10` | -373.5 | -395.2 | +22 | 100% | 100% | 100% | 100% | 1.000 | 1.000 | 57.1 | 56.7 |
| `knn_k15` | -350.1 | -391.6 | +42 | 100% | 100% | 100% | 100% | 1.000 | 1.000 | 44.2 | 44.1 |
| `disk_R60` | -275.3 | -277.7 | +2 | 0% | 0% | 0% | 0% | 0.227 | 0.221 | 3658.8 | 3663.5 |
| `disk_R120` | -397.8 | -412.7 | +15 | 58% | 83% | 59% | 83% | 0.962 | 0.986 | 587.4 | 239.0 |
| `er_p0.2` | -1129.9 | -1210.6 | +81 | 80% | 80% | 80% | 80% | 0.675 | 0.725 | 255.9 | 227.9 |
| `er_p0.3` | -806.0 | -1019.0 | +213 | 98% | 98% | 98% | 98% | 0.941 | 0.906 | 59.0 | 71.4 |
| `er_p0.5` | -520.2 | -684.1 | +164 | 100% | 100% | 100% | 100% | 1.000 | 0.984 | 42.4 | 42.7 |

## Converged-only metrics

| topology | n dec | n acs | dec R | acs R | d-a | dec op | acs op | dec pos | acs pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fully_connected` | 100 | 100 | -351.7 | -425.3 | +74 | 1.000 | 1.000 | 39.5 | 39.6 |
| `star` | 43 | 100 | -713.0 | -530.5 | -183 | 1.000 | 1.000 | 49.8 | 47.0 |
| `wheel` | 63 | 77 | -793.6 | -832.4 | +39 | 0.988 | 0.999 | 51.9 | 50.7 |
| `binary_tree` | 16 | 51 | -881.0 | -796.9 | -84 | 0.966 | 0.999 | 92.1 | 95.1 |
| `line` | 16 | 48 | -726.6 | -813.2 | +87 | 0.996 | 0.979 | 169.2 | 203.7 |
| `ring_k1` | 9 | 25 | -798.1 | -800.0 | +2 | 0.999 | 0.999 | 105.7 | 114.8 |
| `ring_k2` | 35 | 51 | -799.9 | -842.4 | +42 | 0.999 | 0.999 | 68.1 | 64.3 |
| `ring_k3` | 81 | 69 | -699.3 | -848.1 | +149 | 1.000 | 1.000 | 51.9 | 51.0 |
| `mst` | 70 | 94 | -604.1 | -591.3 | -13 | 0.998 | 0.999 | 213.9 | 217.2 |
| `delaunay` | 42 | 57 | -823.3 | -826.8 | +3 | 0.999 | 0.997 | 90.3 | 90.2 |
| `knn_k5` | 34 | 37 | -445.3 | -417.6 | -28 | 1.000 | 0.998 | 107.2 | 130.0 |
| `knn_k10` | 100 | 100 | -373.5 | -395.2 | +22 | 1.000 | 1.000 | 57.1 | 56.7 |
| `knn_k15` | 100 | 100 | -350.1 | -391.6 | +42 | 1.000 | 1.000 | 44.2 | 44.1 |
| `disk_R60` | 0 | 0 | - | - | - | - | - | - | - |
| `disk_R120` | 58 | 83 | -399.1 | -412.0 | +13 | 1.000 | 1.000 | 41.0 | 41.2 |
| `er_p0.2` | 34 | 41 | -827.5 | -859.7 | +32 | 0.999 | 1.000 | 55.7 | 55.6 |
| `er_p0.3` | 84 | 79 | -707.2 | -873.4 | +166 | 1.000 | 1.000 | 46.8 | 47.7 |
| `er_p0.5` | 100 | 95 | -520.2 | -638.6 | +118 | 1.000 | 1.000 | 42.4 | 42.4 |

## Non-converged-only metrics

| topology | n dec | n acs | dec R | acs R | d-a | dec op | acs op | dec pos | acs pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fully_connected` | 0 | 0 | - | - | - | - | - | - | - |
| `star` | 57 | 0 | -1322.3 | - | - | 0.372 | - | 81.6 | - |
| `wheel` | 37 | 23 | -1400.0 | -1622.8 | +223 | 0.501 | 0.429 | 76.4 | 63.4 |
| `binary_tree` | 84 | 49 | -1222.6 | -1274.3 | +52 | 0.373 | 0.462 | 140.9 | 122.9 |
| `line` | 84 | 52 | -1199.3 | -1214.4 | +15 | 0.403 | 0.531 | 208.5 | 182.2 |
| `ring_k1` | 91 | 75 | -1273.6 | -1348.4 | +75 | 0.333 | 0.439 | 153.3 | 143.9 |
| `ring_k2` | 65 | 49 | -1331.7 | -1476.8 | +145 | 0.467 | 0.452 | 94.6 | 83.4 |
| `ring_k3` | 19 | 31 | -1341.5 | -1498.1 | +157 | 0.442 | 0.508 | 75.0 | 65.0 |
| `mst` | 30 | 6 | -1012.7 | -929.1 | -84 | 0.652 | 0.815 | 263.4 | 211.8 |
| `delaunay` | 58 | 43 | -1326.9 | -1464.0 | +137 | 0.665 | 0.669 | 104.8 | 97.9 |
| `knn_k5` | 66 | 63 | -422.7 | -420.0 | -3 | 0.676 | 0.663 | 2155.4 | 2167.3 |
| `knn_k10` | 0 | 0 | - | - | - | - | - | - | - |
| `knn_k15` | 0 | 0 | - | - | - | - | - | - | - |
| `disk_R60` | 100 | 100 | -275.3 | -277.7 | +2 | 0.227 | 0.221 | 3658.8 | 3663.5 |
| `disk_R120` | 42 | 17 | -396.2 | -416.3 | +20 | 0.910 | 0.918 | 1342.0 | 1204.7 |
| `er_p0.2` | 66 | 59 | -1285.7 | -1454.4 | +169 | 0.508 | 0.534 | 359.0 | 347.6 |
| `er_p0.3` | 16 | 21 | -1324.7 | -1566.8 | +242 | 0.633 | 0.551 | 122.7 | 160.6 |
| `er_p0.5` | 0 | 5 | - | -1548.1 | - | - | 0.675 | - | 47.9 |

## Takeaways

**Decentralized policy beats ACS overall** (reward gap > +10):
- `fully_connected`: overall d-a **+74**, conv-only d-a **+74**, dec cv% **100%** vs acs cv% **100%**
- `ring_k3`: overall d-a **+228**, conv-only d-a **+149**, dec cv% **81%** vs acs cv% **69%**
- `knn_k10`: overall d-a **+22**, conv-only d-a **+22**, dec cv% **100%** vs acs cv% **100%**
- `knn_k15`: overall d-a **+42**, conv-only d-a **+42**, dec cv% **100%** vs acs cv% **100%**
- `disk_R120`: overall d-a **+15**, conv-only d-a **+13**, dec cv% **58%** vs acs cv% **83%**
- `er_p0.2`: overall d-a **+81**, conv-only d-a **+32**, dec cv% **34%** vs acs cv% **41%**
- `er_p0.3`: overall d-a **+213**, conv-only d-a **+166**, dec cv% **84%** vs acs cv% **79%**
- `er_p0.5`: overall d-a **+164**, conv-only d-a **+118**, dec cv% **100%** vs acs cv% **95%**

**Decentralized policy fails vs ACS overall** (reward gap < -10):
- `star`: overall d-a **-530**, conv-only d-a **-183**, dec cv% **43%** vs acs cv% **100%**
- `binary_tree`: overall d-a **-137**, conv-only d-a **-84**, dec cv% **16%** vs acs cv% **51%**
- `line`: overall d-a **-102**, conv-only d-a **+87**, dec cv% **16%** vs acs cv% **48%**
- `ring_k1`: overall d-a **-19**, conv-only d-a **+2**, dec cv% **9%** vs acs cv% **25%**
- `mst`: overall d-a **-115**, conv-only d-a **-13**, dec cv% **70%** vs acs cv% **94%**
- `delaunay`: overall d-a **-15**, conv-only d-a **+3**, dec cv% **42%** vs acs cv% **57%**
- `knn_k5`: overall d-a **-11**, conv-only d-a **-28**, dec cv% **34%** vs acs cv% **37%**

**Convergence rate differs significantly between dec and ACS** (|Δcv%| > 15 points):
- `star`: dec cv% **43%** vs acs cv% **100%** (Δ -57 pts)
- `binary_tree`: dec cv% **16%** vs acs cv% **51%** (Δ -35 pts)
- `line`: dec cv% **16%** vs acs cv% **48%** (Δ -32 pts)
- `ring_k1`: dec cv% **9%** vs acs cv% **25%** (Δ -16 pts)
- `ring_k2`: dec cv% **35%** vs acs cv% **51%** (Δ -16 pts)
- `mst`: dec cv% **70%** vs acs cv% **94%** (Δ -24 pts)
- `disk_R120`: dec cv% **58%** vs acs cv% **83%** (Δ -25 pts)

---
*Generated from `eval_checkpoint_decentralized.py` — 100 episodes, max_steps=2500, num_agents=20, conv window=100, pos_rate=2.0, op_rate=0.05*
