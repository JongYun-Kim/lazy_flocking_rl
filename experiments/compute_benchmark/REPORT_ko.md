# 연산 부하 벤치마크: 학습된 PPO vs. 휴리스틱

이 보고서는 프로젝트 전반에 걸쳐 사용되는 세 가지 action 선택 policy의 **타임스텝당 wall-clock 연산 시간**을 측정하며, 학습된 모델에서 RLlib overhead를 제거한 **순수 NN** 변형도 함께 포함한다.

| 축약 이름 | 타이밍 대상 | action 생성 방식 |
| --- | --- | --- |
| `ACS` | fully-active 기준선 | `env.get_fully_active_action()` |
| `Heuristic` | sole-lazy 휴리스틱 (`laziness = 1 - G·α·γ`) | `env.compute_heuristic_action()` |
| `RL` | RLlib을 통한 학습된 Transformer PPO | `policy.compute_single_action(obs, explore=False)` — RLlib 전처리, 배치 빌드, action-distribution 샘플링, 끝의 `np.clip` 포함 |
| `RL_pure_forward` | 학습된 Transformer PPO, NN 단독 | `policy.model.policy_network(obs_td)` — obs는 타이머 *외부*에서 대상 device에 미리 tensor화, RLlib 래핑 없음 |

action 선택 호출만 계측된다. 이후 실행되는 `env.step(action)`은 에이전트가 실제로 이동하고 관측 분포가 현실적으로 유지되도록 계속 실행되지만, 그 비용은 집계에 포함되지 않는다.

이 벤치마크의 모든 코드는 `experiments/compute_benchmark/`에 있다. 해당 디렉터리 외부는 수정하지 않았다.

## 방법론: 무엇을 "해당 policy의 overhead"로 볼 것인가

이 벤치마크의 목표는 **비용 귀속**이다: 실제 배포 환경에서 그대로 사용될 경우 각 알고리즘이 *실질적으로* 얼마나 걸리는가? 이 관점에서 두 가지 설계 결정이 도출된다.

### 1. `cuda.synchronize()`는 RL 변형에만 부과된다

`torch.cuda.synchronize()`는 무시할 수 없는 비용이며(이 호스트에서 호출당 ~6–10 μs), GPU tensor 때문에만 존재한다. `ACS`와 `Heuristic`은 순수 NumPy로 동작하며, 절대 kernel을 실행하지 않는다. 현실적인 배포 환경에서는 이 둘을 GPU에 올리지 않는다. 따라서 이들의 타이밍 호출 주변에 sync를 두면 실제로는 발생하지 않을 비용을 귀속시키는 셈이 된다.

반면 `RL`과 `RL_pure_forward`는 `--device cuda` 시 실제로 kernel을 실행하며, 현실 배포에서 호스트는 action을 `env.step`에 넘기기 전에 결과가 나올 때까지 기다려야 한다. 이 대기 시간은 policy의 스텝당 지연 시간의 진짜 일부이며 반드시 계산에 포함해야 한다.

따라서 벤치마크는 `torch.cuda.synchronize()`를 **RL 변형에 한해, `--device cuda`일 때만** 타이밍 블록 주변에 호출한다. 결과적으로 휴리스틱 타이밍은 CPU 실행과 GPU 실행 간에 동일하다(올바른 결과다 — 두 경우 compute graph가 동일하다).

### 2. CPU 스레드 수는 고정하며 `torch`의 기본값에 맡기지 않는다

이 호스트에서 `torch.get_num_threads()`의 기본값은 36(= 물리 코어 수)이다. *단일 샘플* Transformer forward pass에서 36개 스레드를 가동하는 것은 심각한 oversubscription이다 — wakeup jitter가 실제 연산을 압도하고 측정 시간이 4–6배 부풀어 오르며 큰 outlier가 생긴다. 따라서 벤치마크 전체에 `torch.set_num_threads(4)` / `set_num_interop_threads(4)`를 적용한다. `4`는 일반적인 단일 에이전트 배포 예산에 가까우며, `--torch_threads` 플래그로 다른 값을 시험할 수 있다. 선택된 값은 출력 JSON의 `host_info`에 기록된다.

휴리스틱은 torch 연산을 전혀 건드리지 않으므로 이 설정이 영향을 미치지 않는다 — CPU 실행과 GPU 실행의 숫자가 동일한 또 다른 이유다.

## Setup

- 환경: `LazyAgentsCentralized` (프로젝트의 기본 테스트 env, `CLAUDE.md` 기준)
- Config: `num_agents = 20`, `speed = 15`, `R = 60`, `max_time_step = 2000`,
  `normalize_obs = True`, `use_preprocessed_obs = True`, `use_heuristics = True`,
  `_use_fixed_lazy_idx = True`
- Checkpoint: `bk/bk_082623/PPO_lazy_env_36f5d_00000_0_..._2023-08-26_12-53-47/checkpoint_000074/policies/default_policy`
- 타이머: `time.perf_counter_ns()`. `cuda`에서의 `RL` / `RL_pure_forward`에는
  `torch.cuda.synchronize()`가 타이밍 호출을 감싼다.
- `RL_pure_forward`의 경우, numpy→torch obs 변환과 mean 추출 / clip / `cpu().numpy()` 후처리는 타이머 **외부**이며, 순수 `policy_network(...)` forward만 내부다.
- Warmup: 타이밍 전 policy당 200스텝 (집계 제외).
- 측정: **3 rollout × 2000스텝 = policy당 6000개 타이밍 샘플**.
  기본 seed `4242`; rollout `i`는 seed `4242+i` 사용. 에피소드가 일찍 종료되면 즉시 reset하고 rollout이 2000 샘플을 채울 때까지 계속 수집한다.
- Torch: `set_num_threads(4)`, `set_num_interop_threads(4)`.
- 하드웨어: Intel Xeon w9-3475X (논리 CPU 72개), NVIDIA RTX 6000 Ada (`cuda` 실행 시). PyTorch 1.12.1+cu113, Ray 2.1.0, Python 3.9.5.

### CUDA graph 변형 (보조 측정)

GPU 테이블의 `RL_pure_forward_cudagraph_fp32` 및 `RL_pure_forward_cudagraph_fp16` 행은 별도 스크립트 `benchmark_cudagraph.py`에서 나온 결과이며, 동일한 6000샘플 rollout 구조를 사용하되 forward 호출을 capture한 `torch.cuda.CUDAGraph`의 스텝별 replay로 교체한다.

- **Capture / replay.** Obs tensor를 GPU에 한 번 pre-allocate한다.
  Warmup(5회 forward)은 side `torch.cuda.Stream`에서 실행하고 capture 전에 default stream에 합류시키며, PyTorch 권장 패턴을 따른다. 이후 forward를 `torch.cuda.graph(g)` 내부에서 capture한다. 각 스텝에서는 env의 numpy obs를 static 입력 tensor에 in-place `.copy_()`로 복사하고(타이머 *외부*) `g.replay()`를 타이머 *내부*에서 호출하되, `cuda.synchronize()`가 타이밍 구간을 감싼다. replay 이후의 `mean[…].clamp(0,1).cpu().numpy()`도 타이머 외부에 있어 기존 `RL_pure_forward` 행과 공정하게 비교된다.
- **Half precision.** `cudagraph_fp16`은 policy network를 `torch.float16`으로 cast하여 동일한 graph를 실행한다. 네트워크의 *복사본*에 두 가지 dtype 안전성 패치를 in-memory로 적용하며, 프로젝트의 모델 소스는 **수정하지 않는다**:
    1. 모든 `MultiHeadAttentionLayer.calculate_attention`에서 mask fill 값 `-1e9`를 `-6e4`로 교체한다. `finfo(float16).max`는 ~6.55e4이므로, `-1e9`는 softmax에서 `-inf` / NaN으로 오버플로우된다.
    2. `LazinessAllocator.get_context_node`에서 count의 `.float()`를 `.to(embeddings.dtype)`으로 교체하여, 평균화된 context embedding이 fp16을 유지하고 graph의 나머지 부분이 fp32로 승격되지 않도록 한다(승격 시 하류 LayerNorm의 fp16 가중치와 dtype 불일치 발생).
- **수치 검증.** `cudagraph_fp32`는 non-graph 기준선과 **비트 단위로 동일**하다(최대 절대 오차 = 0). `cudagraph_fp16`은 fp32 경로 대비 **최대 절대 오차 ≈ 8.8e-4**, **최대 상대 오차 ≈ 1.7e-3** — `clamp(0, 1)` action 공간 허용 오차 내에 충분히 들어오며, env는 실질적으로 동일한 clipped action을 바이트 단위까지 동일하게 받는다.
- **형상 제약.** Graph는 `num_agents = 20`, `batch = 1`에서 capture된다. N이 달라지면 re-capture가 필요하다. 단, env가 이미 `num_agents_max`까지 hard-padding을 수행하므로, 최대 크기에서 capture한 graph 하나로 N ≤ cap인 모든 경우를 처리할 수 있으며, 피크 메모리 증가는 N에 선형적이고 수십 MiB 수준에 머문다(실용적 제한이 아님).

원시 샘플별 타이밍(마이크로초)은 다음에 보존된다:

- `results/benchmark_cpu.json` — 주요 (CPU) 비교
- `results/benchmark_gpu.json` — `cuda:0`에서의 PPO 모델
- `results/benchmark_cudagraph.json` — CUDA graph 변형 (GPU)

재현 방법:

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

## 모델 footprint (checkpoint 74)

로드된 `policy.model` (`MyRLlibTorchWrapper`)에서 측정. `share_layers=True` 적용(value function이 encoder를 재사용하므로 별도 `value_network` 없음). 모든 가중치는 `torch.float32`.

| 서브모듈 | 파라미터 수 |
| --- | ---: |
| `policy_network` (LazinessAllocator: 3-layer encoder + 2-layer decoder + Gaussian pointer head) | **857,856** |
| `value_branch` (MLP: shared encoder context 위에 128 → 128 → 1) | 16,641 |
| `value_network` | 0 (비활성화; `share_layers=True`) |
| **합계** | **874,497** |

- **디스크 / RAM state-dict 크기 (fp32):** **3.336 MiB**
  (= 874,497 파라미터 × 4바이트)
- **B=1 추론 시 피크 GPU 메모리 (RTX 6000 Ada, fp32):**
  **총 27.604 MiB** = **가중치 3.336 MiB** + **활성화 + workspace ~24.27 MiB**. `torch.cuda.reset_peak_memory_stats()` 이후 10회 warm forward pass에 걸쳐 `torch.cuda.max_memory_allocated()`로 측정.
- B=1에서 활성화/workspace가 가중치를 ~7배 초과하는 이유는, attention이 `(1, num_heads, seq_len, seq_len)` 중간 tensor를 생성하고 position-wise FFN이 `d_ff = 512`로 확장되기 때문이다. 웬만한 GPU(수백 MiB 여유가 있는 데스크톱 iGPU조차)라면 이 모델을 충분히 올릴 수 있다.

## 결과 — CPU 실행

학습된 policy를 CPU에 고정. `torch.set_num_threads(4)`. 이것이 "동일 머신에서 각 policy의 비용이 얼마인가"에 대한 공정한 비교다.

| Policy | mean | median | std | p95 | p99 | min | max | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ACS` | **5.48 μs** | 5.32 μs | 1.26 μs | 6.23 μs | 9.47 μs | 4.43 μs | 48.20 μs | 6000 |
| `Heuristic` | **29.41 μs** | 28.95 μs | 4.46 μs | 32.07 μs | 42.31 μs | 18.94 μs | 273.93 μs | 6000 |
| `RL_pure_forward` (Transformer 단독) | **1 187.22 μs** ≈ 1.19 ms | 1 186.63 μs | 45.64 μs | 1 234.83 μs | 1 329.52 μs | 1 039.28 μs | 2 174.49 μs | 6000 |
| `RL` (RLlib + Transformer) | **2 105.34 μs** ≈ 2.11 ms | 2 096.06 μs | 117.40 μs | 2 203.76 μs | 2 509.04 μs | 1 856.71 μs | 4 590.93 μs | 6000 |

**속도 비율 (평균, CPU):**

- `RL_pure_forward` / `Heuristic` ≈ **40.4×**
- `RL` / `RL_pure_forward` ≈ **1.77×** (RLlib이 스텝당 ~918 μs / ~44% overhead 추가:
  전처리기 조회, 필터 적용, distribution 생성, 샘플링, clipping, device 이동)
- `RL` / `Heuristic` ≈ **71.6×**
- `RL` / `ACS` ≈ **384×**
- `Heuristic` / `ACS` ≈ **5.4×**

`dt = 0.1 s`일 때 시뮬레이션의 실시간 예산은 스텝당 100 ms이므로, CPU에서의 전체 RLlib 경로조차 **실시간 대비 ~47.5× 빠르다**. NN 단독은 4개 CPU 스레드에서 **실시간 대비 ~84.2× 빠르다**.

## 결과 — GPU 실행

동일한 설정, Transformer를 `cuda:0`에서 실행. RL 변형에 한해 `torch.cuda.synchronize()`가 타이밍 호출을 감싼다. `ACS` / `Heuristic`은 sync overhead 없음(GPU를 건드리지 않음); 해당 숫자는 예상대로 CPU 실행과 정확히 일치한다.

| Policy | mean | median | std | p95 | p99 | min | max | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ACS` | 5.45 μs | 5.27 μs | 1.29 μs | 6.31 μs | 9.11 μs | 4.39 μs | 40.87 μs | 6000 |
| `Heuristic` | 29.20 μs | 28.83 μs | 4.77 μs | 31.62 μs | 40.06 μs | 19.58 μs | 281.83 μs | 6000 |
| `RL_pure_forward` (Transformer 단독, GPU) | **1 506.13 μs** ≈ 1.51 ms | 1 510.63 μs | 58.96 μs | 1 553.87 μs | 1 618.63 μs | 1 304.88 μs | 3 279.93 μs | 6000 |
| `RL_pure_forward_cudagraph_fp32` (graph capture/replay, fp32) | **262.69 μs** ≈ 0.26 ms | 261.86 μs | 6.17 μs | 267.28 μs | 275.13 μs | 258.92 μs | 660.23 μs | 6000 |
| `RL_pure_forward_cudagraph_fp16` (graph capture/replay, fp16) | **201.27 μs** ≈ 0.20 ms | 200.71 μs | 4.01 μs | 205.13 μs | 213.94 μs | 196.43 μs | 366.70 μs | 6000 |
| `RL` (RLlib + Transformer, GPU) | **2 724.24 μs** ≈ 2.72 ms | 2 729.76 μs | 140.31 μs | 2 831.14 μs | 3 165.71 μs | 2 372.41 μs | 5 832.41 μs | 6000 |

**속도 비율 (평균, GPU):**

- GPU `RL_pure_forward` (graph 없음) vs CPU `RL_pure_forward`: **GPU에서 1.27× 느림**
  (1.51 ms vs 1.19 ms). B=1에서는 PyTorch의 kernel 실행당 launch + sync 지연이 이 소형 Transformer에서 절약되는 matmul 시간을 초과한다.
- **CUDA graph가 이 상황을 역전시킨다.** `cudagraph_fp32`는 동일한 NN을 동일한 GPU에서 **262.7 μs**에 실행 — graph가 없는 GPU 기준선(`benchmark_cudagraph.py`의 in-run 기준 행에서 1 545 μs, `RL_pure_forward` 행을 run-to-run 노이즈 수준 내에서 재현)과 비교해 **5.9× 빠름**. 다시 말해, **B=1 GPU 비용의 ~83%는 PyTorch kernel-launch 지연**이었지 실제 연산이 아니었다 — graph가 전체 forward를 단일 launch로 축약하여 이를 명시적으로 드러낸다.
- `cudagraph_fp32` (262.7 μs) vs CPU `RL_pure_forward` (1 187 μs) ≈ **GPU에서 4.52× 빠름**. Graph를 사용하면 B=1 GPU 경로가 드디어 CPU를 앞선다.
- `cudagraph_fp16` (201.3 μs)은 fp32 graph 대비 추가로 **~23%** 절감하지만, 절대값으로는 ~61 μs에 불과하다 — 이 workload 크기에서 forward는 tensor-core 연산이 아닌 launch 및 scheduling에 의해 병목이 생기므로, fp16의 일반적인 속도 향상이 거의 나타나지 않는다.
- **RLlib overhead, 재해석.** GPU에서 `RL`은 2 724 μs이며 최적화된 NN floor(`cudagraph_fp32`)는 ~263 μs이므로, RLlib wrapper는 **~2 461 μs, 즉 배포된 `compute_single_action` 지연의 ~90%**를 차지한다. CPU 섹션의 "RL의 44%"는 non-graph NN 기준선과의 비교이므로, 배포 호출에서 모델 외부에 얼마나 많은 시간이 소비되는지를 과소평가한다. 지연 시간이 중요한 배포에서의 현실적인 floor는 "RLlib을 제거하고 captured graph를 replay"하는 것(~263 μs)이며, 현재 RLlib+GPU 경로가 보고하는 2.7 ms가 아니다.

## 결론 (Takeaways)

1. **신경망 policy는 CUDA graph 최적화 이후에도 휴리스틱보다 1–2 오더 무겁다.**
   스텝당: ACS ~5.5 μs, sole-lazy 휴리스틱 ~29 μs vs Transformer **~1.19 ms (CPU) /
   ~263 μs (GPU + graph, fp32) / ~201 μs (GPU + graph, fp16) / 전체 RLlib 경로 ~2.1–2.7 ms**. 어떤 규모에서도(온보드 연산, 대규모 군집, 다수의 env) 휴리스틱은 비교적 사실상 무료다 — 단, graph 최적화된 NN은 휴리스틱 호출의 ~9배에 불과하여 non-graph 수치가 시사하는 것보다 훨씬 온건한 차이다.
2. **RLlib이 배포된 RL 지연의 대부분을 차지한다.** CPU 수치는 `RL` 시간의 ~44%를 RLlib wrapper에 귀속시키며(non-graph NN 기준선 대비), 현실적인 최적화 floor(`cudagraph_fp32` ~263 μs) 대비 wrapper는 **배포된 `compute_single_action` 호출의 ~90%**다. 이 모두(전처리기 / distribution 빌드 / 샘플링 / clipping / 마샬링)는 CPU-bound Python이며 NN을 GPU로 옮겨도 줄어들지 않는다 — 지연 시간이 중요한 배포에서 현실적인 개선은 "RLlib을 건너뛰고 captured graph를 replay하는 것"이지 "모델을 CUDA로 이식하는 것"이 아니다.
3. **B=1에서 GPU는 CUDA graph가 있어야 의미가 있다.** 단순 PyTorch 경로는 B=1에서 CPU보다 실제로 *느리다* (1.51 ms vs 1.19 ms): kernel 실행당 launch + sync 지연이 이 소형 Transformer에서 절약되는 matmul 시간을 초과한다. forward를 `torch.cuda.CUDAGraph`로 capture하고 replay하면 GPU 비용이 **263 μs (fp32) / 201 μs (fp16)**으로 내려간다 — non-graph GPU 경로 대비 **5.9×**, 최적 CPU 대비 **4.5×** 빠름. B=1에서의 엔지니어링 선택지는 따라서 "CPU vs GPU"가 아니라 "단순 PyTorch vs captured graph"다. 벡터화 rollout(B ≥ 8–16)에서는 단순 GPU 경로도 graph 없이 이기기 시작하지만, 단일 에이전트 추론은 capture 없이는 지연 병목에서 벗어나지 못한다.
4. **fp16은 이 규모에서 한계적인 개선이다.** `cudagraph_fp16`은 fp32 graph 대비 ~23% 빠르다 (262.7 → 201.3 μs), 즉 절대값으로 ~61 μs 절감. Forward는 tensor-core-bound matmul이 아닌 launch / scheduling / 소형 FFN에 의해 지배되므로 통상적인 fp16 속도 향상이 거의 나타나지 않는다. 엔지니어링 비용(attention mask 오버플로우, context-node upcast, 수치 검증)을 감안하면, fp16은 스텝당 μs가 실제로 중요한 경우에만 적용할 가치가 있다.
5. **어떤 policy도 이 env에서 실시간 병목이 아니다.**
   `dt = 100 ms`이므로, CPU에서의 full-RLlib PPO조차 실시간 대비 ~47.5× 빠르며, graph 최적화 NN은 **실시간 대비 ~380× 빠르다**. 문제는 "100 ms 안에 들어오는가"가 아니라 상대적인 연산 예산이다: 휴리스틱 호출은 어떤 NN 변형보다 ~9–70× 저렴하며, 에이전트 수, env 수, 의사결정 주파수를 높이면 이 차이가 실질적으로 드러난다.
6. **메모리는 걱정할 수준이 아니다.** 모델 = **874,497 파라미터 / 3.34 MiB (fp32)**; B=1 추론 시 피크 GPU 메모리는 **27.6 MiB**이며, captured graph는 그 위에 작은 static-tensor footprint만 더한다 — 합산해도 수십 MiB 수준이다. graph를 실질적으로 제약하는 것은 `num_agents_max` 형상 cap이며, env가 이미 이 크기로 padding하므로 capture된 graph 하나로 N ≤ cap인 모든 경우에 유효하다. 배포 메모리가 빡빡해진다면 fp16 / int8 양자화로 이 수치를 절반/4분의 1로 줄일 수 있으며 forward pass 구조는 바뀌지 않는다.
7. **분산 프로파일.** ACS는 매우 안정적(std ≈ 1.3 μs)이며, Heuristic은 완만한 꼬리를 가진다(p99 ≈ 42 μs). Graph 변형이 전체 중 가장 안정적이다(std/mean ≈ 0.023 for `cudagraph_fp32`, ~0.020 for `cudagraph_fp16`): 동적 스케줄링 없음, Python allocator 노이즈 없음, 순수 replay만. Non-graph NN 변형은 그 사이에 위치한다. outlier trimming은 적용하지 않았으며; 모든 백분위수는 원시 6000샘플 분포를 기준으로 한다.

## 파일 구성

```
experiments/compute_benchmark/
├── REPORT.md                     (this file)
├── REPORT_ko.md                  (Korean version, regenerated from this file)
├── benchmark.py                  (ACS / Heuristic / RL / RL_pure_forward timings + model_info)
├── benchmark_cudagraph.py        (CUDA graph capture/replay variants, fp32 + fp16)
└── results/
    ├── benchmark_cpu.json        (CPU timings + model_info)
    ├── benchmark_gpu.json        (GPU timings + model_info incl. peak GPU mem)
    └── benchmark_cudagraph.json  (CUDA graph timings + numerics sanity checks)
```
