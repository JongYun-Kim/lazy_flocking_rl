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

| Method | Reward (L2) | Convergence time (s) | Steering control cost (L2) |
| --- | ---: | ---: | ---: |
| ACS | -112.77 +/- 47.65 | 65.80 +/- 22.49 | 46.97 +/- 27.19 |
| Heuristic | **-78.66 +/- 24.33** | **51.57 +/- 14.36** | **27.09 +/- 12.12** |
| RL | -83.03 +/- 22.03 | 54.74 +/- 14.45 | 28.29 +/- 9.51 |

### PSO (n = 100)

| Method | Reward (L2) | Convergence time (s) | Steering control cost (L2) |
| --- | ---: | ---: | ---: |
| PSO | **-70.82 +/- 29.86** | 52.52 +/- 30.48 | **18.30 +/- 5.67** |

## Data

```
experiments/performance_benchmark/results/
├── acs.json              (1000 episodes, seeds 1-1000)
├── heuristic.json        (1000 episodes, seeds 1-1000)
├── rl.json               (1000 episodes, seeds 1-1000)
└── pso_seed801-900.json  (100 episodes, seeds 801-900)
```
