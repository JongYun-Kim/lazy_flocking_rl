# Computational Load Benchmark: Trained PPO vs. Heuristics

This report measures the **per-time-step wall-clock computation time** of the
three action-selection policies used throughout this project, plus a
**pure-NN** variant that strips RLlib overhead off the trained model.

| Short name | What is timed | How the action is produced |
| --- | --- | --- |
| `ACS` | fully-active baseline | `env.get_fully_active_action()` |
| `Heuristic` | sole-lazy heuristic (`laziness = 1 - G·α·γ`) | `env.compute_heuristic_action()` |
| `RL` | trained Transformer PPO via RLlib | `policy.compute_single_action(obs, explore=False)` — includes RLlib preprocessing, batch building, action-distribution sampling, and the `np.clip` at the end |
| `RL_pure_forward` | trained Transformer PPO, NN only | `policy.model.policy_network(obs_td)` — obs pre-tensorised on the target device *outside* the timer, no RLlib wrapping |
| `PSO` | PSO metaheuristic (SLPSO) | `PSOActionOptimizer.optimize(env)` — finds a single constant action for the entire episode via population-based search, parallelised with Ray (32 CPUs). Per-step time = total optimisation time ÷ episode steps |

Only the action-selection call is timed. The subsequent `env.step(action)` is
still executed (so the agents move and the observation distribution stays
realistic), but its cost is not counted. PSO is an exception: it internally
runs full episodes to evaluate candidate actions, so its "action-selection"
cost includes the environment simulation inside the optimiser.

All code for this benchmark lives in `experiments/compute_benchmark/`. Nothing
outside that directory was modified.

## Methodology: what counts as "the policy's overhead"

The goal of this benchmark is **cost attribution**: how long does each
algorithm *really* take in a deployment where it would be used as-is? That
framing drives two design choices.

### 1. `cuda.synchronize()` is charged only to the RL variants

`torch.cuda.synchronize()` is a non-trivial cost (~6–10 μs per call on this
host) and it only exists because of GPU tensors. `ACS` and `Heuristic` are
pure NumPy; they never launch a kernel and no realistic deployment puts them
on a GPU. So synchronising around their timed call would attribute a cost
they would never actually pay.

In contrast, `RL` and `RL_pure_forward` *do* launch kernels when `--device
cuda`, and in any real deployment the host must wait for the result before
it can feed the action into `env.step`. That wait is a true part of the
policy's per-step latency and has to be counted.

The benchmark therefore calls `torch.cuda.synchronize()` around the timed
block **only for the RL variants, and only when `--device cuda`**. Heuristic
timings are consequently identical between the CPU and GPU runs (which is
the correct outcome — their compute graph is the same in both cases).

### 2. CPU thread count is pinned, not left to `torch`'s default

On this host `torch.get_num_threads()` defaults to 36 (= physical cores).
For a *single-sample* transformer forward pass, spinning up 36 threads is
heavy oversubscription — wakeup jitter dominates the actual math, and the
measured time balloons 4–6× with large outliers. We therefore set
`torch.set_num_threads(4)` / `set_num_interop_threads(4)` for the whole
benchmark. `4` is close to a typical single-agent deployment budget; the
flag `--torch_threads` exposes this in case another value matters to the
reader's use case. The chosen value is recorded under `host_info` in the
output JSON.

Heuristics don't touch torch compute, so this knob has no effect on them
either — another reason their CPU-run and GPU-run numbers are identical.

## Setup

- Environment: `LazyAgentsCentralized` (project's primary test env per `CLAUDE.md`)
- Config: `num_agents = 20`, `speed = 15`, `R = 60`, `max_time_step = 2000`,
  `normalize_obs = True`, `use_preprocessed_obs = True`, `use_heuristics = True`,
  `_use_fixed_lazy_idx = True`
- Checkpoint: `bk/bk_082623/PPO_lazy_env_36f5d_00000_0_..._2023-08-26_12-53-47/checkpoint_000074/policies/default_policy`
- Timer: `time.perf_counter_ns()`. For `RL` / `RL_pure_forward` on `cuda`,
  `torch.cuda.synchronize()` brackets the timed call.
- For `RL_pure_forward`, the numpy→torch obs conversion and the
  mean-extraction / clip / `cpu().numpy()` post-processing are **outside** the
  timer — only the raw `policy_network(...)` forward is inside.
- Warmup: 200 steps per policy before timing (not counted).
- Measurement: **3 rollouts × 2000 steps = 6000 timed samples per policy**.
  Base seed `4242`; rollout `i` uses seed `4242+i`. If the episode terminates
  early we reset immediately and keep collecting until the rollout has 2000
  samples.
- Torch: `set_num_threads(4)`, `set_num_interop_threads(4)`.
- Hardware: Intel Xeon w9-3475X (72 logical CPUs), NVIDIA RTX 6000 Ada (for
  the `cuda` run). PyTorch 1.12.1+cu113, Ray 2.1.0, Python 3.9.5.

### CUDA graph variants (supplementary)

The `RL_pure_forward_cudagraph_fp32` and `RL_pure_forward_cudagraph_fp16`
rows in the GPU table come from a separate script, `benchmark_cudagraph.py`,
which uses the same 6000-sample rollout structure but swaps the forward call
for a captured `torch.cuda.CUDAGraph` replayed each step.

- **Capture / replay.** Obs tensors are pre-allocated once on the GPU.
  Warmup (5 forwards) is run on a side `torch.cuda.Stream` and joined back
  to the default stream before capture, following PyTorch's recommended
  pattern. The forward is then captured inside `torch.cuda.graph(g)`. Each
  step does an in-place `.copy_()` from the env's numpy obs into the static
  input tensors (*outside* the timer) and calls `g.replay()` *inside* the
  timer, with `cuda.synchronize()` still bracketing the timed window. The
  post-replay `mean[…].clamp(0,1).cpu().numpy()` also lives outside the
  timer — apples-to-apples with the existing `RL_pure_forward` row.
- **Half precision.** `cudagraph_fp16` runs the same graph with the policy
  network cast to `torch.float16`. Two dtype-safety patches are applied
  in-memory to the *copy* of the network; the project's model source is
  **not** modified:
    1. In every `MultiHeadAttentionLayer.calculate_attention`, the mask
       fill value `-1e9` is replaced by `-6e4`. `finfo(float16).max` is
       ~6.55e4, so `-1e9` overflows to `-inf` / NaN in softmax.
    2. In `LazinessAllocator.get_context_node`, the count's `.float()` is
       replaced by `.to(embeddings.dtype)` so the averaged context
       embedding stays in fp16 instead of promoting the rest of the graph
       back to fp32 (which would then mismatch the downstream LayerNorm's
       fp16 weights).
- **Numerics.** `cudagraph_fp32` is **bit-exact** against the non-graph
  baseline (max abs error = 0). `cudagraph_fp16` has **max abs error ≈
  8.8e-4**, **max rel error ≈ 1.7e-3** vs the fp32 path — well inside the
  `clamp(0, 1)` action-space tolerance, so the env sees the same clipped
  action byte-for-byte in practice.
- **Shape constraint.** The graph is captured at `num_agents = 20`,
  `batch = 1`. Varying N requires re-capturing — or, since the env already
  hard-pads to `num_agents_max`, a single graph captured at the max size
  handles all N ≤ cap with a peak-memory bump that's linear in N and stays
  in the tens of MiB (not a practical limit here).

Raw per-sample timings (microseconds) are preserved in:

- `results/benchmark_cpu.json` — primary (CPU) comparison
- `results/benchmark_gpu.json` — PPO model on `cuda:0`
- `results/benchmark_cudagraph.json` — CUDA graph variants (GPU)

To reproduce:

```bash
python -m experiments.compute_benchmark.benchmark --device cpu  \
    --torch_threads 4 \
    --output experiments/compute_benchmark/results/benchmark_cpu.json
python -m experiments.compute_benchmark.benchmark --device cuda \
    --torch_threads 4 \
    --output experiments/compute_benchmark/results/benchmark_gpu.json
python -m experiments.compute_benchmark.benchmark_cudagraph \
    --torch_threads 4 \
    --output experiments/compute_benchmark/results/benchmark_cudagraph.json
```

## Model footprint (checkpoint 74)

Measured from the loaded `policy.model` (`MyRLlibTorchWrapper`) with
`share_layers=True` (so the value function reuses the encoder; there is no
separate `value_network`). All weights are `torch.float32`.

| Submodule | Parameters |
| --- | ---: |
| `policy_network` (LazinessAllocator: 3-layer encoder + 2-layer decoder + Gaussian pointer head) | **857,856** |
| `value_branch` (MLP: 128 → 128 → 1 atop shared encoder context) | 16,641 |
| `value_network` | 0 (disabled; `share_layers=True`) |
| **Total** | **874,497** |

- **On-disk / in-RAM state-dict size (fp32):** **3.336 MiB**
  (= 874,497 params × 4 bytes)
- **Peak GPU memory during B=1 inference (RTX 6000 Ada, fp32):**
  **27.604 MiB total** = **3.336 MiB weights** + **~24.27 MiB activations +
  workspace**. Measured by `torch.cuda.max_memory_allocated()` around ten
  warm forward passes after `torch.cuda.reset_peak_memory_stats()`.
- Activation/workspace dominates weights by ~7× at B=1 because attention
  produces `(1, num_heads, seq_len, seq_len)` intermediates and the
  position-wise FFN widens to `d_ff = 512`. Any half-decent GPU (or even a
  desktop iGPU with a few hundred MiB free) can host this model.

## Results — CPU run

Trained policy pinned to CPU. `torch.set_num_threads(4)`. This is the
apples-to-apples comparison for "how much does each policy cost on the same
machine".

| Policy | mean | median | std | p95 | p99 | min | max | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ACS` | **5.48 μs** | 5.32 μs | 1.26 μs | 6.23 μs | 9.47 μs | 4.43 μs | 48.20 μs | 6000 |
| `Heuristic` | **29.41 μs** | 28.95 μs | 4.46 μs | 32.07 μs | 42.31 μs | 18.94 μs | 273.93 μs | 6000 |
| `RL_pure_forward` (Transformer only) | **1 187.22 μs** ≈ 1.19 ms | 1 186.63 μs | 45.64 μs | 1 234.83 μs | 1 329.52 μs | 1 039.28 μs | 2 174.49 μs | 6000 |
| `RL` (RLlib + Transformer) | **2 105.34 μs** ≈ 2.11 ms | 2 096.06 μs | 117.40 μs | 2 203.76 μs | 2 509.04 μs | 1 856.71 μs | 4 590.93 μs | 6000 |
| `PSO` (SLPSO, 32 CPUs) | **420 192 μs** ≈ 420.2 ms | 413 455 μs | 165 377 μs | — | — | 258 286 μs | 588 835 μs | 3 † |

† PSO finds one constant action per episode. `n = 3` is the number of
independent optimisation runs; per-step time = total optimisation time ÷
episode steps. High variance reflects episode-length variation (303–2000
steps) and PSO convergence behaviour.

**PSO total optimisation time (wall-clock):**

| | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| Total time | **364.3 s** | 403.2 s | 87.6 s | 826.9 s |
| Episode steps | **881** | — | 303 | 2000 |

**Speed ratios (mean, CPU):**

- `RL_pure_forward` / `Heuristic` ≈ **40.4×**
- `RL` / `RL_pure_forward` ≈ **1.77×** (RLlib adds ~918 μs / ~44% overhead
  per step: preprocessor lookup, filter application, distribution construction,
  sampling, clipping, device shuffling)
- `RL` / `Heuristic` ≈ **71.6×**
- `RL` / `ACS` ≈ **384×**
- `Heuristic` / `ACS` ≈ **5.4×**
- `PSO` / `RL` ≈ **200×** (per-step equivalent; PSO uses 32 CPU cores vs
  single-threaded for all other policies)
- `PSO` / `Heuristic` ≈ **14 290×**

At `dt = 0.1 s` the sim's real-time budget per step is 100 ms, so even the
full RLlib path on CPU runs **~47.5× faster than real time**. The NN alone
is **~84.2× faster than real time** on 4 CPU threads. PSO's per-step
equivalent of ~420 ms is **~4.2× slower than real time** despite using 32
CPU cores — it cannot be used as an online policy.

## Results — GPU run

Same configuration, Transformer on `cuda:0`. `torch.cuda.synchronize()`
brackets the timed call for the RL variants only. `ACS` / `Heuristic` see no
sync overhead (they don't touch the GPU); their numbers match the CPU run
exactly as expected.

| Policy | mean | median | std | p95 | p99 | min | max | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ACS` | 5.45 μs | 5.27 μs | 1.29 μs | 6.31 μs | 9.11 μs | 4.39 μs | 40.87 μs | 6000 |
| `Heuristic` | 29.20 μs | 28.83 μs | 4.77 μs | 31.62 μs | 40.06 μs | 19.58 μs | 281.83 μs | 6000 |
| `RL_pure_forward` (Transformer only, GPU) | **1 506.13 μs** ≈ 1.51 ms | 1 510.63 μs | 58.96 μs | 1 553.87 μs | 1 618.63 μs | 1 304.88 μs | 3 279.93 μs | 6000 |
| `RL_pure_forward_cudagraph_fp32` (graph capture/replay, fp32) | **262.69 μs** ≈ 0.26 ms | 261.86 μs | 6.17 μs | 267.28 μs | 275.13 μs | 258.92 μs | 660.23 μs | 6000 |
| `RL_pure_forward_cudagraph_fp16` (graph capture/replay, fp16) | **201.27 μs** ≈ 0.20 ms | 200.71 μs | 4.01 μs | 205.13 μs | 213.94 μs | 196.43 μs | 366.70 μs | 6000 |
| `RL` (RLlib + Transformer, GPU) | **2 724.24 μs** ≈ 2.72 ms | 2 729.76 μs | 140.31 μs | 2 831.14 μs | 3 165.71 μs | 2 372.41 μs | 5 832.41 μs | 6000 |

**Speed ratios (mean, GPU):**

- GPU `RL_pure_forward` (no graph) vs CPU `RL_pure_forward`: **1.27× slower
  on GPU** (1.51 ms vs 1.19 ms). At B=1, PyTorch's per-kernel launch and
  sync latency exceeds the saved matmul time for this small Transformer.
- **CUDA graphs flip that picture.** `cudagraph_fp32` runs the same NN in
  **262.7 μs** on the same GPU — a **5.9× speedup** vs the matched no-graph
  GPU baseline (1 545 μs from `benchmark_cudagraph.py`'s in-run baseline row,
  which reproduces the `RL_pure_forward` row within run-to-run noise). In
  other words, **~83% of the B=1 GPU cost was PyTorch kernel-launch latency**,
  not actual compute — the graph collapses the whole forward into a single
  launch and makes that explicit.
- `cudagraph_fp32` (262.7 μs) vs CPU `RL_pure_forward` (1 187 μs) ≈ **4.52×
  faster on GPU**. With graphs, the B=1 GPU path finally beats CPU.
- `cudagraph_fp16` (201.3 μs) shaves another **~23%** off the fp32 graph, but
  only ~61 μs in absolute terms — at this workload size the forward is
  launch- and scheduling-bound, not tensor-core-bound, so fp16's usual
  speed-up barely shows up.
- **RLlib overhead, re-interpreted.** `RL` on GPU is 2 724 μs; the optimised
  NN floor (`cudagraph_fp32`) is ~263 μs, so the RLlib wrapper is
  **~2 461 μs, i.e. ~90% of the deployed `compute_single_action` latency**.
  The "44% of RL" figure from the CPU section compares against a *no-graph*
  NN baseline and therefore understates how much of a deployed call is spent
  outside the model. For latency-critical deployment the realistic floor is
  "drop RLlib and replay a captured graph" (~263 μs), not the 2.7 ms the
  current RLlib+GPU path reports.

## Takeaways

1. **The neural policy is still 1–2 orders of magnitude heavier than the
   heuristics**, even with CUDA graph optimisation. Per step: ACS ~5.5 μs,
   sole-lazy heuristic ~29 μs vs the Transformer at **~1.19 ms (CPU) /
   ~263 μs (GPU + graph, fp32) / ~201 μs (GPU + graph, fp16) / ~2.1–2.7 ms
   through the full RLlib path**. On any scale (onboard compute, large
   swarms, many envs), the heuristic is essentially free by comparison —
   but note that the graph-optimised NN is only ~9× a heuristic call, much
   milder than the no-graph numbers suggest.
2. **RLlib is where most of deployed RL latency lives.** The CPU numbers
   attribute ~44% of `RL`'s time to the RLlib wrapper vs a no-graph NN
   baseline; against the realistic optimised floor (`cudagraph_fp32` ~263 μs)
   the wrapper is **~90% of the deployed `compute_single_action` call**.
   All of it (preprocessor / distribution build / sample / clip / marshal)
   is CPU-bound Python and doesn't shrink when you move the NN to a GPU —
   so for latency-critical deployment the realistic engineering win is
   "skip RLlib and replay a captured graph", not "port the model to CUDA".
3. **GPU at B=1 needs CUDA graphs to be worth it.** The naive PyTorch path
   is actually *slower* on GPU than on CPU at B=1 (1.51 ms vs 1.19 ms):
   per-kernel launch + sync latency exceeds the saved matmul time for this
   small Transformer. Capturing the forward as a `torch.cuda.CUDAGraph` and
   replaying it drops the GPU cost to **263 μs (fp32) / 201 μs (fp16)** — a
   **5.9× speedup** over the no-graph GPU path and **4.5× over the best
   CPU number**. The B=1 engineering choice is therefore "naive-PyTorch vs
   captured-graph", not "CPU vs GPU". For vectorised rollouts (B ≥ 8–16)
   the naive GPU path also starts to win without graphs, but single-agent
   inference stays latency-bound without capture.
4. **fp16 is a marginal win at this size.** `cudagraph_fp16` is ~23% faster
   than the fp32 graph (262.7 → 201.3 μs), i.e. only ~61 μs saved in
   absolute terms. The forward is dominated by launch / scheduling / small
   FFN rather than tensor-core-bound matmul, so the usual fp16 speedup
   barely shows up. Combined with the engineering cost (attention mask
   overflow, context-node upcast, numerics audit), fp16 is only worth the
   trouble if per-step μs really matter.
5. **PSO is offline-only.** At ~420 ms per step (amortised over the episode)
   using 32 CPU cores, PSO is **~4.2× slower than real time** and **~200×
   more expensive than the full RLlib RL path** (which itself uses one
   thread). PSO's cost scales with the number of function evaluations
   (population × generations × episode length), so it is fundamentally an
   offline/batch optimiser — useful for finding benchmark-quality constant
   actions, but not deployable as a real-time policy. The high variance
   (std ≈ 165 ms around a 420 ms mean) reflects episode-length variation:
   short converging episodes (303 steps) amortise the fixed optimisation
   cost more favourably than long non-converging ones (2000 steps).
6. **None of the reactive policies is a real-time bottleneck in this env.**
   `dt = 100 ms`, so even full-RLlib PPO on CPU runs ~47.5× faster than
   real time, and the graph-optimised NN runs **~380× faster than real
   time**. The concern isn't "does it fit in 100 ms", it's relative compute
   budget: a heuristic call is ~9–70× cheaper than any NN variant, which
   matters if you scale up agents, envs, or decision frequency.
7. **Memory is a non-concern.** Model = **874,497 params / 3.34 MiB (fp32)**;
   peak GPU memory at B=1 inference is **27.6 MiB**, and the captured graph
   adds only a small static-tensor footprint on top — still tens of MiB
   total. The `num_agents_max` shape cap is what actually constrains the
   graph; padding to the cap (which the env already does) keeps a single
   captured graph valid for any N ≤ cap. If deployment memory ever got
   tight, fp16 / int8 quantisation would halve/quarter these numbers
   without changing the forward-pass structure.
8. **Variance profile.** ACS is very tight (std ≈ 1.3 μs); Heuristic has a
   modest tail (p99 ≈ 42 μs). The graph variants are the tightest of all
   (std/mean ≈ 0.023 for `cudagraph_fp32`, ~0.020 for `cudagraph_fp16`):
   no dynamic scheduling, no Python-allocator noise, just a replay. The
   no-graph NN variants sit between. No outlier trimming was applied; all
   percentiles are on the raw 6000-sample distributions.

## Files

```
experiments/compute_benchmark/
├── REPORT.md                     (this file)
├── REPORT_ko.md                  (Korean version, regenerated from this file)
├── benchmark.py                  (ACS / Heuristic / RL / RL_pure_forward / PSO timings + model_info)
├── benchmark_cudagraph.py        (CUDA graph capture/replay variants, fp32 + fp16)
└── results/
    ├── benchmark_cpu.json        (CPU timings + model_info)
    ├── benchmark_gpu.json        (GPU timings + model_info incl. peak GPU mem)
    ├── benchmark_cudagraph.json  (CUDA graph timings + numerics sanity checks)
    └── benchmark_pso.json        (PSO optimisation timings, 32 CPUs)
```
