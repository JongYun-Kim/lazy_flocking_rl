# 연산 부하 벤치마크: 학습된 PPO vs. 휴리스틱

이 보고서는 본 프로젝트에서 사용되는 세 가지 행동 선택 정책과, 학습된 모델에서
RLlib 오버헤드를 제거한 **순수 NN** 변형의 **타임스텝당 벽시계 연산 시간**을
측정한다.

| 약칭 | 측정 대상 | 행동 생성 방식 |
| --- | --- | --- |
| `ACS` | 완전 활성 기준선 | `env.get_fully_active_action()` |
| `Heuristic` | 단일 게으름 휴리스틱 (`laziness = 1 - G·α·γ`) | `env.compute_heuristic_action()` |
| `RL` | RLlib 경유 학습된 Transformer PPO | `policy.compute_single_action(obs, explore=False)` — RLlib 전처리, 배치 구성, 행동 분포 샘플링, 마지막 `np.clip` 포함 |
| `RL_pure_forward` | 학습된 Transformer PPO, NN만 | `policy.model.policy_network(obs_td)` — 관측값을 대상 디바이스에서 미리 텐서화한 후 타이머 *밖에서* 처리, RLlib 래핑 없음 |
| `PSO` | PSO 메타휴리스틱 (SLPSO) | `PSOActionOptimizer.optimize(env)` — 집단 기반 탐색을 통해 에피소드 전체에 대한 단일 상수 행동을 찾으며, Ray(32 CPU)로 병렬화. 스텝당 시간 = 총 최적화 시간 ÷ 에피소드 스텝 수 |

행동 선택 호출만 시간을 측정한다. 이후의 `env.step(action)`은 여전히 실행되어
(에이전트가 이동하고 관측 분포가 현실적으로 유지되도록) 하지만, 그 비용은 포함하지
않는다. PSO는 예외로, 후보 행동을 평가하기 위해 내부적으로 전체 에피소드를
실행하므로 "행동 선택" 비용에 최적화기 내부의 환경 시뮬레이션이 포함된다.

이 벤치마크의 모든 코드는 `experiments/compute_benchmark/`에 있다. 해당 디렉터리
외부의 파일은 수정되지 않았다.

## 방법론: "정책의 오버헤드"로 간주하는 것

이 벤치마크의 목표는 **비용 귀속**이다: 각 알고리즘을 현재 상태 그대로 배포할 때
*실제로* 얼마나 걸리는가? 이 관점이 두 가지 설계 선택을 이끈다.

### 1. `cuda.synchronize()`는 RL 변형에만 부과

`torch.cuda.synchronize()`는 무시할 수 없는 비용(이 호스트에서 호출당 ~6–10 μs)이며
GPU 텐서 때문에만 존재한다. `ACS`와 `Heuristic`은 순수 NumPy이며 커널을 실행하지
않고, 현실적인 배포 환경에서 GPU에 올리는 경우도 없다. 따라서 이들의 측정 구간
주위에 동기화를 걸면 실제로는 절대 발생하지 않을 비용을 귀속시키게 된다.

반면 `RL`과 `RL_pure_forward`는 `--device cuda`일 때 커널을 *실제로* 실행하며,
실제 배포에서도 호스트가 결과를 받아 `env.step`에 전달하기 위해 반드시 대기해야
한다. 이 대기는 정책의 스텝당 지연 시간의 실제 구성 요소이므로 반드시 측정해야
한다.

따라서 벤치마크는 **RL 변형에 한해, `--device cuda`일 때만** 측정 구간 주위에
`torch.cuda.synchronize()`를 호출한다. 결과적으로 휴리스틱 측정값은 CPU 실행과
GPU 실행에서 동일한데, 이는 올바른 결과다 — 그들의 연산 그래프는 양쪽 모두 동일하기
때문이다.

### 2. CPU 스레드 수는 `torch` 기본값이 아닌 고정값 사용

이 호스트에서 `torch.get_num_threads()`의 기본값은 36(= 물리 코어 수)이다.
*단일 샘플* Transformer 순전파에 36개 스레드를 가동하면 심한 과다할당이 발생하여
— 깨움 지터(wakeup jitter)가 실제 연산을 압도하고, 측정 시간이 큰 이상치와 함께
4–6배로 부풀어 오른다. 따라서 벤치마크 전체에 `torch.set_num_threads(4)` /
`set_num_interop_threads(4)`를 설정한다. `4`는 일반적인 단일 에이전트 배포 예산에
가까우며, `--torch_threads` 플래그를 통해 다른 값이 필요한 경우 변경할 수 있다.
선택된 값은 출력 JSON의 `host_info`에 기록된다.

휴리스틱은 torch 연산을 사용하지 않으므로 이 설정은 영향을 미치지 않는다 —
CPU 실행과 GPU 실행의 수치가 동일한 또 다른 이유이다.

## 설정

- 환경: `LazyAgentsCentralized` (`CLAUDE.md` 기준 프로젝트의 주요 테스트 환경)
- 구성: `num_agents = 20`, `speed = 15`, `R = 60`, `max_time_step = 2000`,
  `normalize_obs = True`, `use_preprocessed_obs = True`, `use_heuristics = True`,
  `_use_fixed_lazy_idx = True`
- 체크포인트: `bk/bk_082623/PPO_lazy_env_36f5d_00000_0_..._2023-08-26_12-53-47/checkpoint_000074/policies/default_policy`
- 타이머: `time.perf_counter_ns()`. `RL` / `RL_pure_forward`가 `cuda`일 경우
  `torch.cuda.synchronize()`가 측정 호출을 감싼다.
- `RL_pure_forward`의 경우, numpy→torch 관측값 변환과
  평균 추출 / clip / `cpu().numpy()` 후처리는 타이머 **밖에** 있으며
  — 순수 `policy_network(...)` 순전파만 타이머 안에 있다.
- 워밍업: 정책별 200스텝 (측정에 포함되지 않음).
- 측정: **3회 롤아웃 × 2000스텝 = 정책당 6000개 측정 샘플**.
  기본 시드 `4242`; 롤아웃 `i`는 시드 `4242+i` 사용. 에피소드가 조기 종료되면
  즉시 리셋하고 롤아웃이 2000개 샘플을 확보할 때까지 계속 수집.
- Torch: `set_num_threads(4)`, `set_num_interop_threads(4)`.
- 하드웨어: Intel Xeon w9-3475X (72 논리 CPU), NVIDIA RTX 6000 Ada (`cuda`
  실행용). PyTorch 1.12.1+cu113, Ray 2.1.0, Python 3.9.5.

### CUDA 그래프 변형 (보충)

GPU 표의 `RL_pure_forward_cudagraph_fp32` 및 `RL_pure_forward_cudagraph_fp16`
행은 별도의 스크립트 `benchmark_cudagraph.py`에서 측정한 것으로, 동일한 6000샘플
롤아웃 구조를 사용하되 순전파 호출을 캡처된 `torch.cuda.CUDAGraph`의 리플레이로
대체한다.

- **캡처 / 리플레이.** 관측 텐서는 GPU에 한 번 사전 할당된다.
  워밍업(5회 순전파)은 별도의 `torch.cuda.Stream`에서 실행되고, PyTorch의
  권장 패턴에 따라 캡처 전에 기본 스트림에 합류한다. 이후 순전파가
  `torch.cuda.graph(g)` 내부에서 캡처된다. 매 스텝마다 환경의 numpy 관측값에서
  정적 입력 텐서로의 인플레이스 `.copy_()`가 타이머 *밖에서* 수행되고,
  `g.replay()`가 타이머 *안에서* 호출되며, `cuda.synchronize()`는 여전히
  측정 구간을 감싼다. 리플레이 후의
  `mean[…].clamp(0,1).cpu().numpy()` 역시 타이머 밖에 있어
  — 기존 `RL_pure_forward` 행과 동일한 조건이다.
- **반정밀도.** `cudagraph_fp16`은 정책 네트워크를 `torch.float16`으로 캐스팅한
  동일 그래프를 실행한다. 네트워크의 *사본*에 대해 두 가지 dtype 안전 패치가
  메모리 내에서 적용되며, 프로젝트의 모델 소스는 **수정되지 않는다**:
    1. 모든 `MultiHeadAttentionLayer.calculate_attention`에서 마스크
       채움값 `-1e9`가 `-6e4`로 교체된다. `finfo(float16).max`는
       ~6.55e4이므로, `-1e9`는 softmax에서 `-inf` / NaN으로 오버플로된다.
    2. `LazinessAllocator.get_context_node`에서 카운트의 `.float()`가
       `.to(embeddings.dtype)`로 교체되어, 평균 컨텍스트 임베딩이 fp32로
       승격되어 하위 LayerNorm의 fp16 가중치와 불일치하는 대신 fp16을
       유지한다.
- **수치 정밀도.** `cudagraph_fp32`는 비그래프 기준선 대비 **비트 단위 완전 일치**
  (최대 절대 오차 = 0)이다. `cudagraph_fp16`은 fp32 경로 대비
  **최대 절대 오차 ≈ 8.8e-4**, **최대 상대 오차 ≈ 1.7e-3** — `clamp(0, 1)`
  행동 공간 허용 범위 안에 충분히 들어오므로, 환경은 실질적으로 동일한 클리핑된
  행동을 바이트 단위로 동일하게 받는다.
- **형상 제약.** 그래프는 `num_agents = 20`, `batch = 1`에서 캡처된다.
  N을 변경하려면 재캡처가 필요하다 — 또는 환경이 이미 `num_agents_max`까지
  하드 패딩하므로, 최대 크기에서 캡처된 단일 그래프가 N ≤ cap인 모든 경우를
  처리하며, 피크 메모리 증가는 N에 선형적이고 수십 MiB 수준에 머무른다
  (여기서는 실질적인 제한이 아니다).

스텝별 원시 측정값(마이크로초)은 다음 파일에 보존된다:

- `results/benchmark_cpu.json` — 주요 (CPU) 비교
- `results/benchmark_gpu.json` — PPO 모델을 `cuda:0`에서 실행
- `results/benchmark_cudagraph.json` — CUDA 그래프 변형 (GPU)

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

## 모델 풋프린트 (checkpoint 74)

로드된 `policy.model` (`MyRLlibTorchWrapper`)에서 `share_layers=True`로 측정
(따라서 가치 함수가 인코더를 재사용하며 별도의 `value_network`는 없음). 모든
가중치는 `torch.float32`이다.

| 서브모듈 | 파라미터 수 |
| --- | ---: |
| `policy_network` (LazinessAllocator: 3-layer encoder + 2-layer decoder + Gaussian pointer head) | **857,856** |
| `value_branch` (MLP: 128 → 128 → 1 atop shared encoder context) | 16,641 |
| `value_network` | 0 (비활성; `share_layers=True`) |
| **합계** | **874,497** |

- **디스크/메모리 내 state-dict 크기 (fp32):** **3.336 MiB**
  (= 874,497 params × 4 bytes)
- **B=1 추론 시 피크 GPU 메모리 (RTX 6000 Ada, fp32):**
  **27.604 MiB 합계** = **3.336 MiB 가중치** + **~24.27 MiB 활성화 +
  작업 공간**. `torch.cuda.reset_peak_memory_stats()` 후 10회의 웜 순전파
  주위에서 `torch.cuda.max_memory_allocated()`로 측정.
- 활성화/작업 공간이 B=1에서 가중치 대비 ~7배를 차지하는데, 어텐션이
  `(1, num_heads, seq_len, seq_len)` 중간값을 생성하고 위치별 FFN이
  `d_ff = 512`로 확장되기 때문이다. 수백 MiB 여유가 있는 적당한 GPU
  (또는 데스크톱 iGPU)라면 이 모델을 충분히 호스팅할 수 있다.

## 결과 — CPU 실행

학습된 정책을 CPU에 고정. `torch.set_num_threads(4)`. "동일 머신에서 각 정책의
비용이 얼마인가"에 대한 동등 조건 비교이다.

| 정책 | mean | median | std | p95 | p99 | min | max | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ACS` | **5.48 μs** | 5.32 μs | 1.26 μs | 6.23 μs | 9.47 μs | 4.43 μs | 48.20 μs | 6000 |
| `Heuristic` | **29.41 μs** | 28.95 μs | 4.46 μs | 32.07 μs | 42.31 μs | 18.94 μs | 273.93 μs | 6000 |
| `RL_pure_forward` (Transformer만) | **1 187.22 μs** ≈ 1.19 ms | 1 186.63 μs | 45.64 μs | 1 234.83 μs | 1 329.52 μs | 1 039.28 μs | 2 174.49 μs | 6000 |
| `RL` (RLlib + Transformer) | **2 105.34 μs** ≈ 2.11 ms | 2 096.06 μs | 117.40 μs | 2 203.76 μs | 2 509.04 μs | 1 856.71 μs | 4 590.93 μs | 6000 |
| `PSO` (SLPSO, 32 CPU) | **420 192 μs** ≈ 420.2 ms | 413 455 μs | 165 377 μs | — | — | 258 286 μs | 588 835 μs | 3 † |

† PSO는 에피소드당 하나의 상수 행동을 찾는다. `n = 3`은 독립적인 최적화 실행
횟수이며, 스텝당 시간 = 총 최적화 시간 ÷ 에피소드 스텝 수이다. 높은 분산은
에피소드 길이 변동(303–2000스텝)과 PSO 수렴 특성을 반영한다.

**PSO 총 최적화 시간 (벽시계):**

| | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| 총 시간 | **364.3 s** | 403.2 s | 87.6 s | 826.9 s |
| 에피소드 스텝 수 | **881** | — | 303 | 2000 |

**속도 비율 (평균, CPU):**

- `RL_pure_forward` / `Heuristic` ≈ **40.4×**
- `RL` / `RL_pure_forward` ≈ **1.77×** (RLlib이 스텝당 ~918 μs / ~44% 오버헤드
  추가: 전처리기 조회, 필터 적용, 분포 구성, 샘플링, 클리핑, 디바이스 전송)
- `RL` / `Heuristic` ≈ **71.6×**
- `RL` / `ACS` ≈ **384×**
- `Heuristic` / `ACS` ≈ **5.4×**
- `PSO` / `RL` ≈ **200×** (스텝당 환산 기준; PSO는 32 CPU 코어를 사용하며
  다른 모든 정책은 단일 스레드)
- `PSO` / `Heuristic` ≈ **14 290×**

`dt = 0.1 s`에서 시뮬레이션의 실시간 스텝 예산은 100 ms이므로, CPU에서의 전체
RLlib 경로도 **실시간 대비 ~47.5배 빠르다**. NN 단독으로는 4개 CPU 스레드에서
**실시간 대비 ~84.2배 빠르다**. PSO의 스텝당 환산값 ~420 ms는 32 CPU 코어를
사용함에도 **실시간 대비 ~4.2배 느려** — 온라인 정책으로 사용할 수 없다.

## 결과 — GPU 실행

동일한 구성, Transformer를 `cuda:0`에서 실행. RL 변형에 한해
`torch.cuda.synchronize()`가 측정 호출을 감싼다. `ACS` / `Heuristic`은
동기화 오버헤드가 없다(GPU를 사용하지 않음). 이들의 수치는 예상대로 CPU 실행과
정확히 일치한다.

| 정책 | mean | median | std | p95 | p99 | min | max | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ACS` | 5.45 μs | 5.27 μs | 1.29 μs | 6.31 μs | 9.11 μs | 4.39 μs | 40.87 μs | 6000 |
| `Heuristic` | 29.20 μs | 28.83 μs | 4.77 μs | 31.62 μs | 40.06 μs | 19.58 μs | 281.83 μs | 6000 |
| `RL_pure_forward` (Transformer만, GPU) | **1 506.13 μs** ≈ 1.51 ms | 1 510.63 μs | 58.96 μs | 1 553.87 μs | 1 618.63 μs | 1 304.88 μs | 3 279.93 μs | 6000 |
| `RL_pure_forward_cudagraph_fp32` (그래프 캡처/리플레이, fp32) | **262.69 μs** ≈ 0.26 ms | 261.86 μs | 6.17 μs | 267.28 μs | 275.13 μs | 258.92 μs | 660.23 μs | 6000 |
| `RL_pure_forward_cudagraph_fp16` (그래프 캡처/리플레이, fp16) | **201.27 μs** ≈ 0.20 ms | 200.71 μs | 4.01 μs | 205.13 μs | 213.94 μs | 196.43 μs | 366.70 μs | 6000 |
| `RL` (RLlib + Transformer, GPU) | **2 724.24 μs** ≈ 2.72 ms | 2 729.76 μs | 140.31 μs | 2 831.14 μs | 3 165.71 μs | 2 372.41 μs | 5 832.41 μs | 6000 |

**속도 비율 (평균, GPU):**

- GPU `RL_pure_forward` (그래프 미사용) vs CPU `RL_pure_forward`: **GPU에서
  1.27배 느림** (1.51 ms vs 1.19 ms). B=1에서 PyTorch의 커널별 실행 및 동기화
  지연이 이 작은 Transformer의 절약된 행렬곱 시간을 초과한다.
- **CUDA 그래프가 이 상황을 뒤집는다.** `cudagraph_fp32`는 동일 GPU에서 동일 NN을
  **262.7 μs**에 실행 — 대응하는 비그래프 GPU 기준선(1 545 μs,
  `benchmark_cudagraph.py`의 실행 내 기준선 행에서 측정되며 `RL_pure_forward` 행을
  실행 간 노이즈 범위 내에서 재현) 대비 **5.9배 속도 향상**. 다시 말해,
  **B=1 GPU 비용의 ~83%는 PyTorch 커널 실행 지연**이지 실제 연산이 아니었으며
  — 그래프가 전체 순전파를 단일 실행으로 통합하여 이를 명시적으로 보여준다.
- `cudagraph_fp32` (262.7 μs) vs CPU `RL_pure_forward` (1 187 μs) ≈ **GPU에서
  4.52배 빠름**. 그래프를 사용하면 B=1 GPU 경로가 비로소 CPU를 이긴다.
- `cudagraph_fp16` (201.3 μs)은 fp32 그래프 대비 추가로 **~23%** 절감하지만,
  절대값으로는 ~61 μs에 불과 — 이 워크로드 크기에서 순전파는 텐서 코어가 아닌
  실행 및 스케줄링에 병목되므로, fp16의 통상적인 속도 향상이 거의 나타나지 않는다.
- **RLlib 오버헤드, 재해석.** GPU에서 `RL`은 2 724 μs; 최적화된 NN 하한
  (`cudagraph_fp32`)은 ~263 μs이므로, RLlib 래퍼가
  **~2 461 μs, 즉 배포된 `compute_single_action` 지연의 ~90%**를 차지한다.
  CPU 섹션의 "RL의 44%" 수치는 *비그래프* NN 기준선과의 비교이므로, 배포된 호출
  중 모델 외부에서 소비되는 비율을 과소평가한다. 지연 시간이 중요한 배포에서의
  현실적인 하한은 "RLlib을 제거하고 캡처된 그래프를 리플레이"(~263 μs)하는 것이지,
  현재 RLlib+GPU 경로가 보고하는 2.7 ms가 아니다.

## 시사점

1. **신경망 정책은 여전히 휴리스틱 대비 1–2자릿수 더 무겁다** — CUDA 그래프
   최적화를 적용하더라도. 스텝당: ACS ~5.5 μs, 단일 게으름 휴리스틱 ~29 μs vs
   Transformer **~1.19 ms (CPU) / ~263 μs (GPU + 그래프, fp32) /
   ~201 μs (GPU + 그래프, fp16) / ~2.1–2.7 ms (전체 RLlib 경로)**. 어떤
   규모에서든(온보드 컴퓨팅, 대규모 군집, 다수 환경) 휴리스틱은 비교 시 사실상
   무료이다 — 다만 그래프 최적화된 NN은 휴리스틱 호출의 ~9배에 불과하여,
   비그래프 수치가 시사하는 것보다 훨씬 완만하다.
2. **배포된 RL 지연의 대부분은 RLlib에 있다.** CPU 수치는 비그래프 NN 기준선 대비
   `RL` 시간의 ~44%를 RLlib 래퍼에 귀속시키지만, 현실적인 최적화 하한
   (`cudagraph_fp32` ~263 μs) 기준으로는 래퍼가
   **배포된 `compute_single_action` 호출의 ~90%**를 차지한다. 이 모두
   (전처리기 / 분포 구성 / 샘플 / 클립 / 마샬링)는 CPU 바운드 Python이며
   NN을 GPU로 옮겨도 줄어들지 않는다 — 따라서 지연 시간이 중요한 배포에서의
   현실적인 엔지니어링 개선은 "RLlib을 건너뛰고 캡처된 그래프를 리플레이"하는
   것이지, "모델을 CUDA로 포팅"하는 것이 아니다.
3. **B=1에서 GPU가 가치를 가지려면 CUDA 그래프가 필요하다.** 단순 PyTorch 경로는
   실제로 B=1에서 CPU보다 GPU가 *더 느리다* (1.51 ms vs 1.19 ms): 커널별 실행 +
   동기화 지연이 이 작은 Transformer의 절약된 행렬곱 시간을 초과한다. 순전파를
   `torch.cuda.CUDAGraph`로 캡처하여 리플레이하면 GPU 비용이
   **263 μs (fp32) / 201 μs (fp16)**로 떨어진다 — 비그래프 GPU 경로 대비
   **5.9배 속도 향상**, **최적 CPU 수치 대비 4.5배**. 따라서 B=1 엔지니어링
   선택은 "CPU vs GPU"가 아니라 "단순 PyTorch vs 캡처된 그래프"이다.
   벡터화된 롤아웃(B ≥ 8–16)에서는 단순 GPU 경로도 그래프 없이 이기기 시작하지만,
   단일 에이전트 추론은 캡처 없이는 지연에 묶여 있다.
4. **fp16은 이 크기에서 한계적인 개선이다.** `cudagraph_fp16`은 fp32 그래프 대비
   ~23% 빠르며 (262.7 → 201.3 μs), 절대값으로는 ~61 μs 절감에 불과하다.
   순전파는 텐서 코어 바운드 행렬곱이 아닌 실행 / 스케줄링 / 작은 FFN에 의해
   지배되므로, 통상적인 fp16 속도 향상이 거의 나타나지 않는다. 엔지니어링 비용
   (어텐션 마스크 오버플로, 컨텍스트 노드 업캐스트, 수치 감사)을 감안하면,
   fp16은 스텝당 μs가 정말로 중요한 경우에만 수고할 가치가 있다.
5. **PSO는 오프라인 전용이다.** 32 CPU 코어를 사용하여 스텝당 ~420 ms(에피소드
   기준 분할)에서, PSO는 **실시간 대비 ~4.2배 느리고** 전체 RLlib RL 경로
   (단일 스레드 사용) 대비 **~200배 더 비싸다**. PSO의 비용은 함수 평가 횟수
   (집단 크기 × 세대 수 × 에피소드 길이)에 비례하므로, 근본적으로 오프라인/배치
   최적화기이다 — 벤치마크 품질의 상수 행동을 찾는 데는 유용하지만, 실시간
   정책으로 배포할 수는 없다. 높은 분산(평균 420 ms 주위에서 std ≈ 165 ms)은
   에피소드 길이 변동을 반영한다: 짧은 수렴 에피소드(303스텝)는 고정된 최적화
   비용을 더 유리하게 분할하고, 긴 비수렴 에피소드(2000스텝)는 그렇지 않다.
6. **반응형 정책 중 이 환경에서 실시간 병목이 되는 것은 없다.**
   `dt = 100 ms`이므로, CPU에서의 전체 RLlib PPO도 실시간 대비 ~47.5배 빠르게
   실행되고, 그래프 최적화된 NN은 **실시간 대비 ~380배 빠르다**. 걱정되는 것은
   "100 ms 안에 들어가는가"가 아니라 상대적 연산 예산이다: 휴리스틱 호출은
   어떤 NN 변형보다 ~9–70배 저렴하며, 에이전트, 환경, 또는 의사결정 빈도를
   확장할 때 이것이 중요해진다.
7. **메모리는 문제가 되지 않는다.** 모델 = **874,497 params / 3.34 MiB (fp32)**;
   B=1 추론 시 피크 GPU 메모리는 **27.6 MiB**이며, 캡처된 그래프는 그 위에
   작은 정적 텐서 풋프린트만 추가 — 총합은 여전히 수십 MiB 수준이다.
   `num_agents_max` 형상 상한이 그래프를 실제로 제약하며, 상한까지 패딩하면
   (환경이 이미 수행) 단일 캡처 그래프가 N ≤ cap인 모든 경우에 유효하다.
   배포 메모리가 부족해지면, fp16 / int8 양자화가 이 수치를 반/4분의 1로
   줄이되 순전파 구조는 변경하지 않는다.
8. **분산 프로파일.** ACS는 매우 촘촘하다 (std ≈ 1.3 μs); Heuristic은
   완만한 꼬리를 가진다 (p99 ≈ 42 μs). 그래프 변형이 가장 촘촘하다
   (std/mean ≈ 0.023 `cudagraph_fp32`, ~0.020 `cudagraph_fp16`):
   동적 스케줄링 없음, Python 할당기 노이즈 없음, 오직 리플레이뿐이다.
   비그래프 NN 변형은 그 사이에 위치한다. 이상치 트리밍은 적용되지 않았으며,
   모든 백분위수는 원시 6000샘플 분포 기준이다.

## 파일

```
experiments/compute_benchmark/
├── REPORT.md                     (영문 원본)
├── REPORT_ko.md                  (한국어 버전, 이 파일)
├── benchmark.py                  (ACS / Heuristic / RL / RL_pure_forward / PSO 측정 + model_info)
├── benchmark_cudagraph.py        (CUDA 그래프 캡처/리플레이 변형, fp32 + fp16)
└── results/
    ├── benchmark_cpu.json        (CPU 측정값 + model_info)
    ├── benchmark_gpu.json        (GPU 측정값 + model_info, 피크 GPU 메모리 포함)
    ├── benchmark_cudagraph.json  (CUDA 그래프 측정값 + 수치 정합성 검사)
    └── benchmark_pso.json        (PSO 최적화 측정값, 32 CPU)
```
