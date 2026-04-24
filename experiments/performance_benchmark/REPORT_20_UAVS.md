# Performance Benchmark: 20 UAVs

## Objective

Compare four laziness-allocation strategies for a 20-UAV flocking task under
the quadratic cost metric J (see paper for full formulation). Lower J is
better. The two components of J are the steering control cost C (sum of
squared heading-rate inputs, normalized per agent) and the convergence time
penalty η·t_f.

## Methods

| Method | Description |
| --- | --- |
| ACS | Fully-active baseline — all agents apply full control at every step |
| Heuristic | Sole-lazy heuristic — one agent is designated lazy per step via a graph-based rule |
| RL | Transformer-based PPO policy (checkpoint 74) trained to allocate laziness |
| PSO | SLPSO metaheuristic — searches for the optimal constant laziness vector per episode, minimising J directly (deterministic seed control) |

## Conditions

- **Swarm:** 20 UAVs, fully-connected communication topology
- **Episodes:** 1000 (seeds 1–1000) for ACS, Heuristic, RL; 100 (seeds 801–900) for PSO
- **Dynamics:** v = 15 m/s, interaction radius R = 60 m, dt = 0.1 s, T_max = 1000 steps (100 s)
- **Convergence criteria:** spatial entropy < 45, velocity entropy < 0.1, and their rates of change below threshold
- **PSO:** deterministic (per-episode seed propagated to internal RNG)

## Results

### ACS, Heuristic, RL (n = 1000)

| Method | J | C (steering) | t_f (s) | Order φ |
| --- | ---: | ---: | ---: | ---: |
| ACS | 138.83 ± 66.25 | 73.03 ± 46.81 | 65.80 ± 22.48 | 0.9959 ± 0.0375 |
| Heuristic | **88.83 ± 31.37** | 37.26 ± 20.29 | **51.57 ± 14.35** | 0.9998 ± 0.0060 |
| RL | 91.72 ± 25.30 | **36.98 ± 14.14** | 54.74 ± 14.45 | **1.0000 ± 0.0001** |

### PSO (n = 100)

| Method | J | C (steering) | t_f (s) | Order φ |
| --- | ---: | ---: | ---: | ---: |
| PSO | **73.96 ± 23.53** | **15.64 ± 12.53** | 58.31 ± 32.72 | 0.7173 ± 0.3811 |

Order parameter φ = √(1 − (σ_vel / v)²), measured at the final time step.
Values near 1.0 indicate well-aligned velocities.

**PSO convergence note:** 64 of 100 PSO episodes converge within T_max; the
remaining 36 reach the time horizon with fully-lazy allocation (C = 0,
t_f = T_max = 100 s). Under the quadratic cost, PSO discovers that for
certain initial conditions the time-penalty alone (J = 100) is cheaper than
any non-trivial constant laziness vector. These 36 cases correspond to
difficult initial conditions where ACS incurs J = 204.6 on average;
Heuristic (J = 91.8) and RL (J = 94.2), which adapt their allocation at
every step, still converge on these cases.

### Statistical tests (paired t-test, same initial conditions)

Comparisons involving PSO use the 100 overlapping seeds (801–900).

**Episode cost J:**

| A vs B | n | mean diff | t | p | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic vs ACS | 1000 | −50.00 | −22.76 | 1.0e-92 | −0.720 |
| RL vs ACS | 1000 | −47.11 | −21.35 | 1.2e-83 | −0.676 |
| PSO vs ACS | 100 | −59.53 | −11.77 | 1.6e-20 | −1.183 |
| Heuristic vs RL | 1000 | −2.89 | −2.37 | 1.8e-02 | −0.075 |
| PSO vs Heuristic | 100 | −11.80 | −4.21 | 5.7e-05 | −0.423 |
| PSO vs RL | 100 | −13.81 | −4.52 | 1.7e-05 | −0.454 |

**Convergence time t_f:**

| A vs B | n | mean diff (s) | t | p | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic vs ACS | 1000 | −14.23 | −17.57 | 2.0e-60 | −0.556 |
| RL vs ACS | 1000 | −11.07 | −13.19 | 1.0e-36 | −0.417 |
| PSO vs ACS | 100 | −4.29 | −2.25 | 2.7e-02 | −0.226 |
| Heuristic vs RL | 1000 | −3.17 | −5.11 | 3.9e-07 | −0.162 |
| PSO vs Heuristic | 100 | +7.53 | +2.26 | 2.6e-02 | +0.227 |
| PSO vs RL | 100 | +5.83 | +1.71 | 9.0e-02 | +0.172 |

**Steering control cost C:**

| A vs B | n | mean diff | t | p | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic vs ACS | 1000 | −35.77 | −23.26 | 5.6e-96 | −0.736 |
| RL vs ACS | 1000 | −36.05 | −23.68 | 9.8e-99 | −0.749 |
| PSO vs ACS | 100 | −55.24 | −9.79 | 3.1e-16 | −0.984 |
| Heuristic vs RL | 1000 | +0.28 | +0.37 | 7.1e-01 | +0.012 |
| PSO vs Heuristic | 100 | −19.33 | −9.88 | 2.0e-16 | −0.993 |
| PSO vs RL | 100 | −19.64 | −10.31 | 2.3e-17 | −1.036 |

### Summary

All methods significantly outperform ACS on J (p < 1e-20, |d| > 0.67).
Heuristic and RL achieve comparable steering costs (p = 0.71, not
significant), with Heuristic converging slightly faster (−3.2 s, p < 1e-6).
PSO achieves the lowest J (74.0) and C (15.6), but 36% of episodes do not
converge (φ = 0.72 vs φ ≥ 0.996 for the other methods).

## Data

```
experiments/performance_benchmark/results/
├── acs.json        (1000 episodes, seeds 1-1000)
├── heuristic.json  (1000 episodes, seeds 1-1000)
├── rl.json         (1000 episodes, seeds 1-1000)
└── pso.json        (100 episodes, seeds 801-900)
```
