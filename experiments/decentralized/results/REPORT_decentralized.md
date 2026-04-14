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
| `fully_connected` | -251.8 | -251.7 | -329.4 | +78 | +78 | -0 | 100% | 100% | 100% | 1.000 | 1.000 | 1.000 | 39.5 | 39.5 | 39.6 |
| `star` | -706.8 | -335.2 | -419.2 | -288 | +84 | -372 | 100% | 100% | 100% | 0.517 | 0.997 | 0.993 | 75.8 | 41.1 | 47.7 |
| `wheel` | -730.0 | -461.4 | -752.7 | +23 | +291 | -269 | 100% | 100% | 100% | 0.640 | 0.964 | 0.782 | 66.6 | 53.8 | 55.5 |
| `binary_tree` | -730.3 | -444.8 | -707.3 | -23 | +263 | -286 | 100% | 100% | 100% | 0.405 | 0.776 | 0.626 | 133.4 | 185.2 | 115.1 |
| `line` | -721.6 | -482.5 | -723.4 | +2 | +241 | -239 | 100% | 100% | 100% | 0.431 | 0.709 | 0.591 | 195.1 | 359.2 | 191.6 |
| `ring_k1` | -769.6 | -486.0 | -793.8 | +24 | +308 | -284 | 100% | 100% | 100% | 0.351 | 0.725 | 0.500 | 146.6 | 242.5 | 130.0 |
| `ring_k2` | -750.7 | -478.8 | -805.9 | +55 | +327 | -272 | 100% | 100% | 100% | 0.569 | 0.825 | 0.615 | 85.8 | 123.9 | 76.0 |
| `ring_k3` | -608.1 | -458.7 | -739.6 | +132 | +281 | -149 | 100% | 100% | 100% | 0.832 | 0.915 | 0.743 | 59.0 | 69.9 | 58.3 |
| `mst` | -525.5 | -426.7 | -487.7 | -38 | +61 | -99 | 100% | 100% | 100% | 0.759 | 0.796 | 0.913 | 242.5 | 357.7 | 228.2 |
| `delaunay` | -749.5 | -513.2 | -775.5 | +26 | +262 | -236 | 100% | 100% | 100% | 0.683 | 0.840 | 0.764 | 105.1 | 114.6 | 95.1 |
| `knn_k5` | -325.7 | -344.5 | -324.2 | -1 | -20 | +19 | 30% | 21% | 35% | 0.771 | 0.681 | 0.775 | 886.2 | 901.1 | 839.8 |
| `knn_k10` | -272.4 | -267.9 | -300.1 | +28 | +32 | -5 | 100% | 100% | 100% | 1.000 | 1.000 | 1.000 | 57.0 | 57.1 | 57.0 |
| `knn_k15` | -249.1 | -248.2 | -296.9 | +48 | +49 | -1 | 100% | 100% | 100% | 1.000 | 1.000 | 1.000 | 44.1 | 44.0 | 44.2 |
| `disk_R60` | -175.5 | -173.8 | -178.2 | +3 | +4 | -2 | 0% | 0% | 0% | 0.226 | 0.240 | 0.220 | 2202.9 | 2190.0 | 2208.1 |
| `disk_R120` | -299.2 | -288.1 | -311.9 | +13 | +24 | -11 | 54% | 8% | 88% | 0.965 | 0.854 | 0.992 | 342.8 | 922.7 | 117.2 |
| `er_p0.2` | -738.1 | -470.8 | -798.4 | +60 | +328 | -267 | 76% | 76% | 76% | 0.566 | 0.883 | 0.642 | 188.2 | 223.2 | 183.1 |
| `er_p0.3` | -610.2 | -451.5 | -761.2 | +151 | +310 | -159 | 98% | 98% | 98% | 0.823 | 0.958 | 0.743 | 65.2 | 68.6 | 63.7 |
| `er_p0.5` | -401.1 | -358.5 | -552.3 | +151 | +194 | -43 | 100% | 100% | 100% | 0.990 | 0.997 | 0.953 | 42.7 | 42.5 | 43.2 |

## Information loss analysis (decentralized vs centralized)

This table isolates the effect of restricting each agent's observation to its local neighbors. Negative d-c means local info hurts; zero means no difference (FC or very dense topologies).

| topology | d-c reward | dec op | cen op | op drop | dec pos | cen pos | pos increase |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fully_connected` | -0.1 | 1.000 | 1.000 | -0.000 | 39.5 | 39.5 | -0.0 |
| `star` | -371.6 | 0.517 | 0.997 | -0.480 | 75.8 | 41.1 | +34.7 |
| `wheel` | -268.5 | 0.640 | 0.964 | -0.324 | 66.6 | 53.8 | +12.9 |
| `binary_tree` | -285.5 | 0.405 | 0.776 | -0.371 | 133.4 | 185.2 | -51.7 |
| `line` | -239.0 | 0.431 | 0.709 | -0.279 | 195.1 | 359.2 | -164.1 |
| `ring_k1` | -283.6 | 0.351 | 0.725 | -0.374 | 146.6 | 242.5 | -95.9 |
| `ring_k2` | -271.8 | 0.569 | 0.825 | -0.256 | 85.8 | 123.9 | -38.1 |
| `ring_k3` | -149.3 | 0.832 | 0.915 | -0.084 | 59.0 | 69.9 | -10.9 |
| `mst` | -98.7 | 0.759 | 0.796 | -0.037 | 242.5 | 357.7 | -115.2 |
| `delaunay` | -236.3 | 0.683 | 0.840 | -0.156 | 105.1 | 114.6 | -9.5 |
| `knn_k5` | +18.8 | 0.771 | 0.681 | +0.090 | 886.2 | 901.1 | -14.9 |
| `knn_k10` | -4.6 | 1.000 | 1.000 | +0.000 | 57.0 | 57.1 | -0.1 |
| `knn_k15` | -0.9 | 1.000 | 1.000 | +0.000 | 44.1 | 44.0 | +0.1 |
| `disk_R60` | -1.7 | 0.226 | 0.240 | -0.014 | 2202.9 | 2190.0 | +12.9 |
| `disk_R120` | -11.1 | 0.965 | 0.854 | +0.112 | 342.8 | 922.7 | -579.9 |
| `er_p0.2` | -267.3 | 0.566 | 0.883 | -0.317 | 188.2 | 223.2 | -35.0 |
| `er_p0.3` | -158.7 | 0.823 | 0.958 | -0.134 | 65.2 | 68.6 | -3.4 |
| `er_p0.5` | -42.6 | 0.990 | 0.997 | -0.007 | 42.7 | 42.5 | +0.1 |

## Takeaways

**Decentralized policy beats ACS** (reward gap > +10, comparable flock rate):
- `fully_connected`: dec-acs **+78**, info loss (dec-cen) **-0**, dec op **1.000**
- `wheel`: dec-acs **+23**, info loss (dec-cen) **-269**, dec op **0.640**
- `ring_k1`: dec-acs **+24**, info loss (dec-cen) **-284**, dec op **0.351**
- `ring_k2`: dec-acs **+55**, info loss (dec-cen) **-272**, dec op **0.569**
- `ring_k3`: dec-acs **+132**, info loss (dec-cen) **-149**, dec op **0.832**
- `delaunay`: dec-acs **+26**, info loss (dec-cen) **-236**, dec op **0.683**
- `knn_k10`: dec-acs **+28**, info loss (dec-cen) **-5**, dec op **1.000**
- `knn_k15`: dec-acs **+48**, info loss (dec-cen) **-1**, dec op **1.000**
- `er_p0.2`: dec-acs **+60**, info loss (dec-cen) **-267**, dec op **0.566**
- `er_p0.3`: dec-acs **+151**, info loss (dec-cen) **-159**, dec op **0.823**
- `er_p0.5`: dec-acs **+151**, info loss (dec-cen) **-43**, dec op **0.990**

**Decentralized policy fails vs ACS** (worse reward or flock breakup):
- `star`: dec-acs **-288**, dec 1f% **100%** vs acs **100%**
- `binary_tree`: dec-acs **-23**, dec 1f% **100%** vs acs **100%**
- `mst`: dec-acs **-38**, dec 1f% **100%** vs acs **100%**
- `disk_R120`: dec-acs **+13**, dec 1f% **54%** vs acs **88%**

**Significant information loss** (dec-cen reward gap < -20):
- `star`: dec-cen **-372**, dec op **0.517** vs cen op **0.997**
- `wheel`: dec-cen **-269**, dec op **0.640** vs cen op **0.964**
- `binary_tree`: dec-cen **-286**, dec op **0.405** vs cen op **0.776**
- `line`: dec-cen **-239**, dec op **0.431** vs cen op **0.709**
- `ring_k1`: dec-cen **-284**, dec op **0.351** vs cen op **0.725**
- `ring_k2`: dec-cen **-272**, dec op **0.569** vs cen op **0.825**
- `ring_k3`: dec-cen **-149**, dec op **0.832** vs cen op **0.915**
- `mst`: dec-cen **-99**, dec op **0.759** vs cen op **0.796**
- `delaunay`: dec-cen **-236**, dec op **0.683** vs cen op **0.840**
- `er_p0.2`: dec-cen **-267**, dec op **0.566** vs cen op **0.883**
- `er_p0.3`: dec-cen **-159**, dec op **0.823** vs cen op **0.958**
- `er_p0.5`: dec-cen **-43**, dec op **0.990** vs cen op **0.997**

---
*Generated from `eval_checkpoint_decentralized.py` — 400 episodes, max_steps=1500, num_agents=20*
