# Performance Benchmark: 20 UAVs

## Objective

Compare the performance of four action-selection methods on the
`LazyAgentsCentralized` flocking environment with 20 UAVs.

## Methods

| Method | Description |
| --- | --- |
| ACS | Fully-active baseline (`action = ones`) |
| Heuristic | Sole-lazy heuristic (`laziness = 1 - G * alpha * gamma`, 1 lazy agent) |
| RL | Trained Transformer PPO (checkpoint 74) |
| PSO | SLPSO metaheuristic (constant action per episode, L2-optimised) |

## Conditions

- **Environment:** `LazyAgentsCentralized`, 20 UAVs, fully-connected topology
- **Episodes:** 1000 (seeds 1--1000) for ACS, Heuristic, RL; 100 (seeds 801--900) for PSO
- **Config:** `speed=15`, `R=60`, `max_time_step=1000`, `dt=0.1`,
  `incomplete_episode_penalty=0`, `std_pos_converged=45`,
  `std_vel_converged=0.1`
- **Reward:** L2 norm for control cost
- **Steering control cost:** `-(episode_reward + convergence_time)`

## Results

### ACS, Heuristic, RL (n = 1000)

| Method | Reward (L2) | Convergence time (s) | Steering control cost (L2) | Order parameter |
| --- | ---: | ---: | ---: | ---: |
| ACS | -112.77 +/- 47.65 | 65.80 +/- 22.49 | 46.97 +/- 27.19 | 0.9959 +/- 0.0376 |
| Heuristic | **-78.66 +/- 24.33** | **51.57 +/- 14.36** | **27.09 +/- 12.12** | 0.9998 +/- 0.0060 |
| RL | -83.03 +/- 22.03 | 54.74 +/- 14.45 | 28.29 +/- 9.51 | **1.0000 +/- 0.0001** |

### PSO (n = 100)

| Method | Reward (L2) | Convergence time (s) | Steering control cost (L2) | Order parameter |
| --- | ---: | ---: | ---: | ---: |
| PSO | **-70.82 +/- 29.86** | 52.52 +/- 30.48 | **18.30 +/- 5.67** | 0.9803 +/- 0.1240 |

Order parameter = `sqrt(1 - (std_vel / speed)^2)`, measured at the last
time step of each episode. Values near 1.0 indicate well-aligned velocities.

### Statistical tests (paired t-test, same initial conditions)

**Reward (L2):**

| A vs B | n | mean diff | t | p | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic vs ACS | 1000 | +34.11 | 21.23 | 7.9e-83 | 0.671 |
| RL vs ACS | 1000 | +29.74 | 18.21 | 3.4e-64 | 0.576 |
| PSO vs ACS | 100 | +37.10 | 13.09 | 2.5e-23 | 1.309 |
| Heuristic vs RL | 1000 | +4.37 | 4.41 | 1.1e-05 | 0.140 |
| PSO vs Heuristic | 100 | +5.82 | 1.65 | 1.0e-01 | 0.165 |
| PSO vs RL | 100 | +8.66 | 2.61 | 1.1e-02 | 0.261 |

**Convergence time:**

| A vs B | n | mean diff (s) | t | p | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic vs ACS | 1000 | -14.23 | -17.57 | 2.0e-60 | -0.556 |
| RL vs ACS | 1000 | -11.07 | -13.19 | 1.0e-36 | -0.417 |
| PSO vs ACS | 100 | -10.09 | -5.99 | 3.4e-08 | -0.599 |
| Heuristic vs RL | 1000 | -3.17 | -5.11 | 3.9e-07 | -0.161 |
| PSO vs Heuristic | 100 | +1.74 | 0.52 | 6.1e-01 | 0.052 |
| PSO vs RL | 100 | +0.04 | 0.01 | 9.9e-01 | 0.001 |

**Steering control cost (L2):**

| A vs B | n | mean diff | t | p | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic vs ACS | 1000 | -19.88 | -22.10 | 2.0e-88 | -0.699 |
| RL vs ACS | 1000 | -18.67 | -20.87 | 1.5e-80 | -0.660 |
| PSO vs ACS | 100 | -27.02 | -9.48 | 1.5e-15 | -0.948 |
| Heuristic vs RL | 1000 | -1.21 | -2.60 | 9.5e-03 | -0.082 |
| PSO vs Heuristic | 100 | -7.57 | -8.18 | 9.6e-13 | -0.818 |
| PSO vs RL | 100 | -8.70 | -8.26 | 6.7e-13 | -0.826 |

All methods significantly outperform ACS (p < 1e-8, medium-to-large effect).
Heuristic vs RL is statistically significant (p < 0.01) but the effect size
is small (|d| < 0.17). PSO vs Heuristic reward difference is not significant
(p = 0.10), but PSO has significantly lower steering control cost (p < 1e-12,
d = -0.82).

## Data

```
experiments/performance_benchmark/results/
├── acs.json              (1000 episodes, seeds 1-1000)
├── heuristic.json        (1000 episodes, seeds 1-1000)
├── rl.json               (1000 episodes, seeds 1-1000)
└── pso_seed801-900.json  (100 episodes, seeds 801-900)
```
