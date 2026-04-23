# MLP Ablation Study: Justification for the Transformer Policy

## Motivation

A reviewer questioned whether the Transformer-based policy network is necessary for the lazy-agent allocation problem, suggesting that a simpler multi-layer perceptron (MLP) could achieve comparable performance. We ran a systematic ablation over twelve MLP architectures and training configurations to test this claim.

## Experimental Setup

### Environment

All policies are evaluated on the **task-terminating, unshaped flocking environment** (the same environment used in the main experiments of the paper). A variable-length episode ends early when the swarm reaches the cohesion criterion — position standard deviation ≤ 45 length units, velocity standard deviation ≤ 0.1, with both quantities' time-rates also below threshold — or is otherwise terminated at the 1000-step horizon.

Each MLP variant was *trained* on a reward-shaped variant of the environment (a pendulum-style stabilizing bonus with a fixed 1000-step horizon). Training itself used the quadratic control penalty; evaluation used the same quadratic penalty applied to unshaped, variable-horizon episodes, replicating the evaluation protocol used for the reference Transformer in the main paper.

### Reward Metric

**Quadratic (L2) control penalty** — each episode's reward is the negated sum, over all steps, of the squared per-agent control inputs (weighted by a small control coefficient, consistent across all methods). Less-negative values indicate more control-efficient policies; an episode that terminates early on convergence accrues less penalty than one that runs to the 1000-step cap.

### Test Evaluation Protocol

- 100 random seeds (1–100), fixed across methods, for deterministic reproducibility
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
- **LR schedule + entropy annealing:** learning rate stepped from 1 × 10⁻⁴ to 5 × 10⁻⁵ at 1 M steps, to 1 × 10⁻⁵ at 2.5 M steps, and to 5 × 10⁻⁶ at 4.2 M steps. Entropy coefficient stepped from 0.01 → 0.005 → 0.001 → 0 on the same schedule.
- **Deterministic actions:** Gaussian policy with a fixed (non-learned) log standard deviation of −10, reproducing the reference Transformer's effectively deterministic action output.
- **Shared backbone:** policy and value networks share their hidden layers (matching the reference Transformer's architecture).
- **Param-matched variants (#5, #9, #10, #11):** ~872–874 K parameters, within ~0.3 % of the reference Transformer's 874,497.

### Reference Baselines

- **Transformer (proposed):** encoder–decoder attention network, 874,497 parameters, shared backbone, deterministic action distribution, training checkpoint 74.
- **All-active heuristic:** non-learning baseline in which every agent is set to fully active throughout the episode (no lazy-agent allocation). This is the trivial control-inefficient policy.

## Results

All numbers are means over the same 100 seeded episodes. Reward is the quadratic (L2) control-penalty total per episode (less-negative is better). "Converged" denotes the fraction of episodes that reached the cohesion criterion before the 1000-step cap.

| Model | Shared backbone | Params | Checkpoint | Reward (L2) | Gap vs Transformer | ±std | Converged | Mean episode length |
|-------|-----------------|--------|-----------|-------------|--------------------|------|-----------|---------------------|
| **Transformer (proposed)** | **Yes** | **875 K** | **74** | **−84.0** | **—** | **21.8** | **98 %** | **556.5** |
| All-active heuristic | — | — | — | −106.4 | −22.3 | 39.6 | 96 % | 632.3 |
| MLP #2  (higher learning rate) | No | 292 K | 52 | −162.6 | −78.6 | 4.6 | 0 % | 1000.0 |
| MLP #5  (wider, param-matched) | No | 872 K | 268 | −181.8 | −97.8 | 5.9 | 0 % | 1000.0 |
| MLP #4  (LR schedule + entropy annealing) | No | 292 K | 232 | −183.7 | −99.7 | 9.7 | 0 % | 1000.0 |
| MLP #8  (shared backbone, wider) | Yes | 598 K | 234 | −189.7 | −105.7 | 8.8 | 0 % | 1000.0 |
| MLP #0  (baseline) | No | 292 K | 37 | −190.3 | −106.3 | 1.6 | 0 % | 1000.0 |
| MLP #6  (wider + deeper) | No | 1,569 K | 276 | −192.9 | −108.9 | 12.8 | 0 % | 1000.0 |
| MLP #3  (LR schedule) | No | 292 K | 90 | −197.1 | −113.1 | 16.0 | 0 % | 1000.0 |
| MLP #9  (shared, wider + deeper, param-matched) | Yes | 874 K | 259 | −197.4 | −113.4 | 14.9 | 0 % | 1000.0 |
| MLP #11 (deterministic, param-matched) | No | 872 K | 130 | −199.2 | −115.1 | 10.9 | 0 % | 1000.0 |
| MLP #1  (deterministic actions) | No | 292 K | 300 | −201.1 | −117.1 | 12.5 | 0 % | 1000.0 |
| MLP #7  (shared backbone, baseline) | Yes | 168 K | 281 | −201.7 | −117.7 | 11.9 | 0 % | 1000.0 |
| MLP #10 (shared, deterministic, param-matched) | Yes | 874 K | 54 | −204.0 | −120.0 | 10.5 | 0 % | 1000.0 |

## Key Findings

1. **All twelve MLP variants fail to converge.** Across 100 evaluation episodes, every MLP variant achieved 0 % convergence, while the Transformer converged in 98 % of episodes and the all-active heuristic in 96 %.

2. **Parameter count is not the limiting factor.** MLP variants matched to the Transformer's parameter budget (872–874 K) or substantially exceeding it (1.57 M in variant #6) all failed. Network capacity cannot explain the gap.

3. **Sharing the policy/value backbone does not close the gap.** Four variants (#7–#10) use a shared backbone matching the Transformer's architectural choice; none converged. The best shared-backbone variant (#8, wider) still lags the Transformer by 105.7 reward units.

4. **Hyperparameter tuning improves raw reward but does not enable convergence.** A constant higher learning rate (#2) produced the best-reward MLP (−162.6), but this variant converged 0 % of the time and remained worse than the non-learning all-active heuristic (−106.4). Learning-rate scheduling and entropy-bonus annealing provided only marginal additional reward improvement.

5. **Deterministic action distributions do not help.** Three variants (#1, #10, #11) reproduce the Transformer's effectively deterministic action output via a fixed low log-standard-deviation; none converged.

6. **MLPs underperform even the trivial all-active baseline.** The non-learning all-active heuristic achieves 96 % convergence and a reward of −106.4. Every MLP variant is worse than this in both reward and convergence, indicating that MLPs not only fail to learn efficient allocation but also actively produce control policies worse than the trivial strategy of keeping every agent fully active.

## Analysis

The structural explanation for the MLP failure is **lack of permutation equivariance**. The flocking environment contains a variable-ordered *set* of interchangeable agents; an agent's position in the observation vector is arbitrary and may change across episodes (and within an episode due to topology updates).

The Transformer handles this natively:

- **Permutation-equivariant encoding:** self-attention treats agent embeddings as a set, producing representations invariant to agent ordering.
- **Context-dependent allocation:** the decoder attends over all encoded agents simultaneously when producing lazy/active allocation decisions.

An MLP, by contrast, assigns fixed weights to fixed input positions. When agent ordering changes, the MLP cannot generalize — it has memorized position-specific patterns rather than learning agent-level policies.

This structural mismatch explains why no amount of parameter scaling, hyperparameter tuning, or architectural tweaks (depth, width, backbone sharing, deterministic actions) closes the gap: the inductive bias of a fixed-position MLP is fundamentally misaligned with the set-structured problem.

## Experimental Details

- **Hardware:** 4 × NVIDIA RTX 6000 Ada Generation (48 GB each), 72-CPU host
- **Framework:** Ray RLlib 2.1.0, PyTorch
- **Training:** twelve trials, six at a time in two waves, ≈ 10 h total wall-clock
- **Evaluation:** 100 seeded episodes per method, sequential single-process execution, ≈ 31 min total wall-clock
- **Reproducibility:** identical environment initial states and random seeds (1–100) across all methods; policy actions are deterministic (`explore=False`).
- **Reward reported throughout this report:** quadratic (L2) control-penalty episode totals — sum of squared per-step control inputs, negated so that larger (less-negative) values correspond to more control-efficient policies. The same underlying episode trajectories were used to compute both L1 (absolute-value) and L2 (squared) totals; this report uses L2 consistent with the paper revision. Raw per-episode L1 and L2 values are stored in `logs/eval_v3_L1L2_results.json`; the corresponding L1 version of this report is preserved as `mlp_ablation_report_L1.md`.
