# Decentralized checkpoint evaluation report

This report compares the centralized checkpoint evaluated in three modes:
- **Decentralized**: each agent observes only its topology neighbors (per-agent local frame, batched forward pass).
- **Centralized**: all agents observe the full swarm (global frame, single forward pass) — the prior evaluation method.
- **ACS**: fully-active ACS baseline (no learned policy).

The FC topology is a sanity check: decentralized and centralized must be identical because every agent sees every other agent.

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

- **reward**: cumulative episode reward over the fixed horizon.
- **1f%** (single flock rate): fraction of episodes where all agents form one connected component at the last step.
- **op** (order parameter): `|mean(v_i)| / speed` — heading alignment (0 = random, 1 = perfectly aligned).
- **pos** (final std_pos): `sqrt(Var(x) + Var(y))` at last step — spatial tightness.
- **d-a**: reward gap between decentralized policy and ACS.
- **c-a**: reward gap between centralized policy and ACS.
- **d-c**: reward gap between decentralized and centralized — quantifies the cost of limiting information to local neighbors.

## Results

| topology | dec R | cen R | acs R | d-a | c-a | d-c | dec 1f% | cen 1f% | acs 1f% | dec op | cen op | acs op | dec pos | cen pos | acs pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fully_connected` | -260.8 | -260.8 | -331.6 | +71 | +71 | +0 | 100% | 100% | 100% | 1.000 | 1.000 | 1.000 | 39.8 | 39.8 | 39.8 |
| `star` | -640.8 | -316.9 | -386.7 | -254 | +70 | -324 | 100% | 100% | 100% | 0.642 | 1.000 | 0.998 | 68.7 | 37.6 | 48.2 |
| `wheel` | -705.1 | -470.4 | -756.5 | +51 | +286 | -235 | 100% | 100% | 100% | 0.636 | 0.969 | 0.805 | 68.6 | 54.7 | 56.0 |
| `binary_tree` | -736.3 | -452.0 | -695.2 | -41 | +243 | -284 | 100% | 100% | 100% | 0.360 | 0.743 | 0.690 | 136.9 | 195.1 | 104.5 |
| `line` | -725.8 | -493.6 | -691.4 | -34 | +198 | -232 | 100% | 100% | 100% | 0.483 | 0.655 | 0.650 | 197.9 | 352.2 | 205.7 |
| `ring_k1` | -771.4 | -507.0 | -784.8 | +13 | +278 | -264 | 100% | 100% | 100% | 0.461 | 0.654 | 0.611 | 134.0 | 246.6 | 130.3 |
| `ring_k2` | -768.7 | -472.0 | -843.7 | +75 | +372 | -297 | 100% | 100% | 100% | 0.533 | 0.837 | 0.621 | 92.7 | 114.8 | 78.1 |
| `ring_k3` | -624.4 | -434.1 | -754.8 | +130 | +321 | -190 | 100% | 100% | 100% | 0.872 | 0.911 | 0.720 | 55.9 | 67.2 | 57.9 |
| `mst` | -525.5 | -411.1 | -516.4 | -9 | +105 | -114 | 100% | 100% | 100% | 0.768 | 0.779 | 0.921 | 245.8 | 387.8 | 218.3 |
| `delaunay` | -737.9 | -519.8 | -808.9 | +71 | +289 | -218 | 100% | 100% | 100% | 0.686 | 0.817 | 0.762 | 108.9 | 117.7 | 95.9 |
| `knn_k5` | -322.1 | -336.6 | -325.7 | +4 | -11 | +15 | 40% | 40% | 40% | 0.813 | 0.820 | 0.755 | 780.2 | 673.1 | 830.4 |
| `knn_k10` | -266.5 | -270.8 | -303.9 | +37 | +33 | +4 | 100% | 100% | 100% | 1.000 | 1.000 | 1.000 | 56.5 | 57.5 | 57.1 |
| `knn_k15` | -245.7 | -252.9 | -295.8 | +50 | +43 | +7 | 100% | 100% | 100% | 1.000 | 1.000 | 1.000 | 44.3 | 44.3 | 44.1 |
| `disk_R60` | -174.5 | -172.5 | -176.9 | +2 | +4 | -2 | 0% | 0% | 0% | 0.203 | 0.220 | 0.208 | 2220.7 | 2207.2 | 2217.9 |
| `disk_R120` | -298.5 | -296.4 | -296.9 | -2 | +0 | -2 | 57% | 13% | 90% | 0.967 | 0.884 | 0.997 | 335.1 | 773.8 | 78.5 |
| `er_p0.2` | -783.2 | -470.9 | -867.6 | +84 | +397 | -312 | 70% | 70% | 70% | 0.437 | 0.852 | 0.460 | 224.8 | 255.3 | 234.3 |
| `er_p0.3` | -597.3 | -429.9 | -794.3 | +197 | +364 | -167 | 97% | 97% | 97% | 0.841 | 0.959 | 0.748 | 64.6 | 64.4 | 67.6 |
| `er_p0.5` | -404.6 | -353.4 | -558.9 | +154 | +205 | -51 | 100% | 100% | 100% | 0.997 | 1.000 | 0.914 | 42.9 | 41.5 | 43.8 |

## Information loss analysis (decentralized vs centralized)

This table isolates the effect of restricting each agent's observation to its local neighbors. Negative d-c means local info hurts; zero means no difference (FC or very dense topologies).

| topology | d-c reward | dec op | cen op | op drop | dec pos | cen pos | pos increase |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fully_connected` | +0.0 | 1.000 | 1.000 | +0.000 | 39.8 | 39.8 | -0.0 |
| `star` | -323.9 | 0.642 | 1.000 | -0.358 | 68.7 | 37.6 | +31.1 |
| `wheel` | -234.7 | 0.636 | 0.969 | -0.333 | 68.6 | 54.7 | +13.9 |
| `binary_tree` | -284.3 | 0.360 | 0.743 | -0.383 | 136.9 | 195.1 | -58.2 |
| `line` | -232.3 | 0.483 | 0.655 | -0.172 | 197.9 | 352.2 | -154.2 |
| `ring_k1` | -264.5 | 0.461 | 0.654 | -0.192 | 134.0 | 246.6 | -112.6 |
| `ring_k2` | -296.7 | 0.533 | 0.837 | -0.304 | 92.7 | 114.8 | -22.1 |
| `ring_k3` | -190.2 | 0.872 | 0.911 | -0.039 | 55.9 | 67.2 | -11.2 |
| `mst` | -114.4 | 0.768 | 0.779 | -0.011 | 245.8 | 387.8 | -142.0 |
| `delaunay` | -218.0 | 0.686 | 0.817 | -0.131 | 108.9 | 117.7 | -8.9 |
| `knn_k5` | +14.6 | 0.813 | 0.820 | -0.007 | 780.2 | 673.1 | +107.1 |
| `knn_k10` | +4.4 | 1.000 | 1.000 | +0.000 | 56.5 | 57.5 | -1.0 |
| `knn_k15` | +7.2 | 1.000 | 1.000 | -0.000 | 44.3 | 44.3 | +0.0 |
| `disk_R60` | -2.0 | 0.203 | 0.220 | -0.016 | 2220.7 | 2207.2 | +13.5 |
| `disk_R120` | -2.1 | 0.967 | 0.884 | +0.083 | 335.1 | 773.8 | -438.7 |
| `er_p0.2` | -312.3 | 0.437 | 0.852 | -0.414 | 224.8 | 255.3 | -30.5 |
| `er_p0.3` | -167.4 | 0.841 | 0.959 | -0.118 | 64.6 | 64.4 | +0.2 |
| `er_p0.5` | -51.1 | 0.997 | 1.000 | -0.003 | 42.9 | 41.5 | +1.4 |

## Takeaways

**Decentralized policy beats ACS** (reward gap > +10, comparable flock rate):
- `fully_connected`: dec-acs **+71**, info loss (dec-cen) **+0**, dec op **1.000**
- `wheel`: dec-acs **+51**, info loss (dec-cen) **-235**, dec op **0.636**
- `ring_k1`: dec-acs **+13**, info loss (dec-cen) **-264**, dec op **0.461**
- `ring_k2`: dec-acs **+75**, info loss (dec-cen) **-297**, dec op **0.533**
- `ring_k3`: dec-acs **+130**, info loss (dec-cen) **-190**, dec op **0.872**
- `delaunay`: dec-acs **+71**, info loss (dec-cen) **-218**, dec op **0.686**
- `knn_k10`: dec-acs **+37**, info loss (dec-cen) **+4**, dec op **1.000**
- `knn_k15`: dec-acs **+50**, info loss (dec-cen) **+7**, dec op **1.000**
- `er_p0.2`: dec-acs **+84**, info loss (dec-cen) **-312**, dec op **0.437**
- `er_p0.3`: dec-acs **+197**, info loss (dec-cen) **-167**, dec op **0.841**
- `er_p0.5`: dec-acs **+154**, info loss (dec-cen) **-51**, dec op **0.997**

**Decentralized policy fails vs ACS** (worse reward or flock breakup):
- `star`: dec-acs **-254**, dec 1f% **100%** vs acs **100%**
- `binary_tree`: dec-acs **-41**, dec 1f% **100%** vs acs **100%**
- `line`: dec-acs **-34**, dec 1f% **100%** vs acs **100%**
- `disk_R120`: dec-acs **-2**, dec 1f% **57%** vs acs **90%**

**Significant information loss** (dec-cen reward gap < -20):
- `star`: dec-cen **-324**, dec op **0.642** vs cen op **1.000**
- `wheel`: dec-cen **-235**, dec op **0.636** vs cen op **0.969**
- `binary_tree`: dec-cen **-284**, dec op **0.360** vs cen op **0.743**
- `line`: dec-cen **-232**, dec op **0.483** vs cen op **0.655**
- `ring_k1`: dec-cen **-264**, dec op **0.461** vs cen op **0.654**
- `ring_k2`: dec-cen **-297**, dec op **0.533** vs cen op **0.837**
- `ring_k3`: dec-cen **-190**, dec op **0.872** vs cen op **0.911**
- `mst`: dec-cen **-114**, dec op **0.768** vs cen op **0.779**
- `delaunay`: dec-cen **-218**, dec op **0.686** vs cen op **0.817**
- `er_p0.2`: dec-cen **-312**, dec op **0.437** vs cen op **0.852**
- `er_p0.3`: dec-cen **-167**, dec op **0.841** vs cen op **0.959**
- `er_p0.5`: dec-cen **-51**, dec op **0.997** vs cen op **1.000**

---
*Generated from `eval_checkpoint_decentralized.py` — 30 episodes, max_steps=1500, num_agents=20*
