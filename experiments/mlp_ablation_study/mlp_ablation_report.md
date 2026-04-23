# MLP Ablation Study: Architectural Attribution of the Transformer Policy

## Motivation

Among the revisions requested by a reviewer, this study evaluates whether a simpler multi-layer perceptron (MLP) policy could replicate the performance of the proposed Transformer policy on the lazy-agent allocation problem. The objective is architectural attribution: to verify that the reported performance gains in the main paper are attributable to the Transformer's inductive bias rather than to the PPO training procedure in general. We compare twelve systematically varied MLP variants against the reference Transformer and the non-learning ACS baseline.

## Experimental Setup

### Environment

All policies are evaluated on the task-terminating, unshaped flocking environment used in the main experiments of the paper. An episode terminates early when the flock satisfies the cohesion criterion defined in the paper, with thresholds:

- spatial entropy ≤ 45 m
- velocity entropy ≤ 0.1 m
- spatial entropy rate ≤ 0.1 m
- velocity entropy rate ≤ 0.2 m

Otherwise the episode is truncated at the 1000-step horizon.

Each MLP variant was *trained* on the shaped-reward training environment defined in the main paper, with a fixed 1000-step horizon.

### Evaluation Metric

We report the per-episode performance metric $J$ defined in the main paper. $J$ combines two components: (i) a **steering control (energy) cost** — the integrated per-agent squared control input, normalized by $v/n$, reflecting the flock's aggregate maneuvering effort; and (ii) a **convergence-time penalty** $\eta\,t_f$ that penalizes slow completion. Both components are positive, and lower $J$ corresponds to a more control-efficient, faster-converging policy.

### Test Evaluation Protocol

- 100 random seeds (1–100), fixed across methods for deterministic reproducibility
- Sequential single-process execution with `explore=False` (deterministic policy actions)
- Identical starting states for every policy at each seed (via per-episode environment deep-copy)

### Training Configuration

All MLP variants were trained with PPO using hyperparameters matched to the reference Transformer (checkpoint 74):

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Parallel rollout workers | 7 |
| Environments per worker | 2 |
| Train batch size | 14,000 |
| SGD minibatch size | 256 |
| SGD iterations per update | 36 |
| Baseline learning rate | 3 × 10⁻⁵ |
| Discount γ | 0.992 |
| GAE λ | 0.96 |
| PPO clip parameter | 0.25 |
| Training iterations | 300 |
| Checkpoint selection | best 32 by training-time eval reward |

### MLP Variants (12 total)

We systematically varied architectural width/depth, policy/value backbone sharing, action-distribution determinism, and optimization hyperparameters:

| # | Variant (paper-facing label) | Hidden layers | Shared backbone | Deterministic actions | Params | Optimization |
|---|------------------------------|---------------|-----------------|-----------------------|--------|--------------|
| 0 | Baseline | [256] × 3 | No | No | 292 K | Baseline HP |
| 1 | Deterministic actions | [256] × 3 | No | Yes (log σ = −10) | 292 K | Baseline HP |
| 2 | Higher learning rate | [256] × 3 | No | No | 292 K | lr = 1 × 10⁻⁴ constant |
| 3 | LR schedule | [256] × 3 | No | No | 292 K | lr 1 × 10⁻⁴ → 5 × 10⁻⁶ |
| 4 | LR schedule + entropy annealing | [256] × 3 | No | No | 292 K | lr schedule + entropy 0.01 → 0 |
| 5 | Wider, param-matched to Transformer | [462] × 3 | No | No | 872 K | lr schedule + entropy annealing |
| 6 | Wider and deeper | [512] × 4 | No | No | 1,569 K | lr schedule + entropy annealing |
| 7 | Shared backbone, baseline | [256] × 3 | Yes | No | 168 K | Baseline HP |
| 8 | Shared backbone, wider | [512] × 3 | Yes | No | 598 K | lr schedule + entropy annealing |
| 9 | Shared backbone, wider + deeper, param-matched | [516] × 4 | Yes | No | 874 K | lr schedule + entropy annealing |
| 10 | Shared backbone, deterministic, param-matched | [516] × 4 | Yes | Yes | 874 K | lr schedule + entropy annealing |
| 11 | Deterministic, param-matched | [462] × 3 | No | Yes | 872 K | lr schedule + entropy annealing |

Legend for optimization schemes:

- **Baseline HP:** constant learning rate of 3 × 10⁻⁵ with no entropy scheduling — matches the reference Transformer exactly.
- **LR schedule + entropy annealing:** learning rate stepped from 1 × 10⁻⁴ → 5 × 10⁻⁵ (1 M steps) → 1 × 10⁻⁵ (2.5 M steps) → 5 × 10⁻⁶ (4.2 M steps); entropy coefficient stepped 0.01 → 0.005 → 0.001 → 0 on the same schedule.
- **Deterministic actions:** Gaussian policy with a fixed (non-learned) log standard deviation of −10, reproducing the reference Transformer's effectively deterministic action output.
- **Shared backbone:** policy and value networks share their hidden layers (matching the reference Transformer's architecture).
- **Param-matched variants (#5, #9, #10, #11):** 872–874 K parameters, within ~0.3 % of the reference Transformer's 874,497.

### Reference Baselines

- **Transformer (proposed):** encoder–decoder attention network, 874,497 parameters, shared backbone, deterministic action distribution, training checkpoint 74.
- **ACS (augmented Cucker-Smale):** the underlying flocking model without the laziness layer — every agent is fully active, no learned allocation.

## Results

All numbers are means over the same 100 seeded episodes. $J$ is the per-episode cost as defined in the main paper (lower is better); "Gap vs Transformer" is $J_{\text{model}} - J_{\text{Transformer}}$, with positive values indicating higher cost. "Converged" is the fraction of episodes that reached the cohesion criterion before the 1000-step cap.

| Model | Shared backbone | Params | Checkpoint | $J$ | Gap vs Transformer | ±std | Converged | Mean episode length (steps) |
|-------|-----------------|--------|-----------|-----|--------------------|------|-----------|------------------------------|
| **Transformer (proposed)** | **Yes** | **875 K** | **74** | **84.0** | **—** | **21.8** | **98 %** | **556.5** |
| ACS | — | — | — | 106.4 | +22.3 | 39.6 | 96 % | 632.3 |
| MLP #2  (higher learning rate) | No | 292 K | 52 | 162.6 | +78.6 | 4.6 | 0 % | 1000.0 |
| MLP #5  (wider, param-matched) | No | 872 K | 268 | 181.8 | +97.8 | 5.9 | 0 % | 1000.0 |
| MLP #4  (LR schedule + entropy annealing) | No | 292 K | 232 | 183.7 | +99.7 | 9.7 | 0 % | 1000.0 |
| MLP #8  (shared backbone, wider) | Yes | 598 K | 234 | 189.7 | +105.7 | 8.8 | 0 % | 1000.0 |
| MLP #0  (baseline) | No | 292 K | 37 | 190.3 | +106.3 | 1.6 | 0 % | 1000.0 |
| MLP #6  (wider + deeper) | No | 1,569 K | 276 | 192.9 | +108.9 | 12.8 | 0 % | 1000.0 |
| MLP #3  (LR schedule) | No | 292 K | 90 | 197.1 | +113.1 | 16.0 | 0 % | 1000.0 |
| MLP #9  (shared, wider + deeper, param-matched) | Yes | 874 K | 259 | 197.4 | +113.4 | 14.9 | 0 % | 1000.0 |
| MLP #11 (deterministic, param-matched) | No | 872 K | 130 | 199.2 | +115.1 | 10.9 | 0 % | 1000.0 |
| MLP #1  (deterministic actions) | No | 292 K | 300 | 201.1 | +117.1 | 12.5 | 0 % | 1000.0 |
| MLP #7  (shared backbone, baseline) | Yes | 168 K | 281 | 201.7 | +117.7 | 11.9 | 0 % | 1000.0 |
| MLP #10 (shared, deterministic, param-matched) | Yes | 874 K | 54 | 204.0 | +120.0 | 10.5 | 0 % | 1000.0 |

For every MLP variant, the mean episode length is exactly the 1000-step cap because not a single evaluation episode triggered the cohesion criterion; the Transformer and ACS, by contrast, terminate early in nearly all episodes.

## Summary of Findings

These are direct observations from the results table.

1. **All twelve MLP variants fail to converge.** Across 100 seeded episodes, every MLP variant achieved 0 % convergence, while the Transformer converged in 98 % of episodes and ACS in 96 %.

2. **Network capacity is not the bottleneck.** MLP variants with parameter counts matched to the Transformer (872–874 K, variants #5, #9, #10, #11) or substantially exceeding it (1.57 M in variant #6) all failed to converge. Capacity does not explain the gap.

3. **Sharing the policy/value backbone does not close the gap.** The four shared-backbone variants (#7–#10) — which mirror the Transformer's architectural choice — all failed. The best of them (#8) still has a cost gap of +105.7 vs. the Transformer.

4. **Hyperparameter tuning reduces cost but does not enable convergence.** The constant higher learning rate (#2) yields the lowest MLP cost ($J = 162.6$), yet still converges 0 % of the time and remains worse than the non-learning ACS baseline ($J = 106.4$). Learning-rate scheduling and entropy-bonus annealing give only marginal further reductions in $J$.

5. **Deterministic action distributions do not help.** Three variants (#1, #10, #11) reproduce the Transformer's effectively deterministic action output via a fixed low log-standard-deviation; none converge.

6. **All MLPs are worse than ACS.** Every MLP variant underperforms the non-learning ACS baseline in both $J$ and convergence rate, indicating that these MLP policies are worse than simply keeping every agent fully active.

## Discussion

A structural explanation consistent with the above pattern is a **lack of permutation equivariance** in the MLP policy. The flocking problem involves a variable-ordered set of interchangeable agents; an agent's index in the observation vector is arbitrary and may change across episodes (and within an episode due to topology updates).

The Transformer handles this natively through its self-attention encoder, which treats the agent embeddings as a set and yields representations invariant to agent ordering; its decoder then attends over all encoded agents simultaneously when producing lazy/active allocation decisions.

An MLP, by contrast, assigns fixed weights to fixed input positions. When agent ordering changes, an MLP has no mechanism to generalize — it must rely on position-indexed patterns rather than agent-level regularities of the problem.

Under this interpretation, the observed insensitivity to parameter count, backbone sharing, optimization-hyperparameter tuning, and action-distribution determinism is expected: none of these modifications alter the inductive bias of position-indexed computation, which is the structural mismatch with a set-structured problem. This is offered as a plausible hypothesis consistent with the empirical findings; direct evidence (e.g., probing the MLP policies with permuted inputs) is not reported here.

## Experimental Details

- **Hardware:** 4 × NVIDIA RTX 6000 Ada Generation (48 GB each), 72-CPU host
- **Framework:** Ray RLlib 2.1.0, PyTorch
- **Training:** twelve trials, six at a time in two waves, ≈ 10 h total wall-clock
- **Evaluation:** 100 seeded episodes per method, sequential single-process execution, ≈ 31 min total wall-clock (restricted to 64 of the host's 72 CPUs)
- **Reproducibility:** identical environment initial states and random seeds (1–100) across all methods; policy actions are deterministic (`explore=False`).
- **Raw data:** per-episode cost sums and episode lengths are stored in `logs/eval_v3_L1L2_results.json`; the per-episode $J$ values reported here correspond to the negated `rewards_L2` entries. An earlier report using the L1-based cost variant is preserved as `mlp_ablation_report_L1.md`.
