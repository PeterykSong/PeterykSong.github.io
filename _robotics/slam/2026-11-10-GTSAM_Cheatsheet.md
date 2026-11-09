---
#order: 70
layout: single
title: "Graph SLAM Optimizer Tutorial, GTSAM Cheatsheet"
date: 2026-11-10 00:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
is_post: True
excerpt: "학습"
topic: slam
related: false

tags:
  - SLAM
  - Graph_SLAM
  - GTSAM
  - G20
  - Robotics  
---

Day 1학습에 대한 요약정리본이다. 

# GTSAM Day 1 치트시트 — 2D Pose Graph SLAM

> 대상: SLAM 기본 개념을 알고 있고 Python으로 GTSAM 최적화 실습을 진행하는 학습자  
> 목표: `Pose2` 기반 Factor Graph를 구성하고, GN/LM으로 풀고, 초기값·가중치·Loop Closure의 영향을 관찰한다.

## 0. 30초 요약

```text
측정값(odometry / loop closure / prior)
          ↓
NonlinearFactorGraph (제약 = Factors)
          +
Values (변수의 초기 추정값)
          ↓
GaussNewtonOptimizer 또는 LevenbergMarquardtOptimizer
          ↓
Values (최적화된 Pose2들)
```

- **Graph**: 어떤 측정 제약이 어떤 상태변수를 연결하는지 정의한다.
- **Values**: 최적화할 상태변수와 초기값을 보관한다.
- **Factor**: 측정 오차와 그 불확실성(noise model)을 표현한다.
- **Prior**: 전체 좌표계의 기준을 고정한다(gauge freedom 제거).
- **Loop closure**: 시간상 떨어진 두 포즈를 연결해 누적 드리프트를 보정한다.
- **Optimizer**: 초기 추정값에서 시작하여 가중 비선형 최소제곱을 푼다.

## 1. 가장 많이 쓰는 Python API

| 목적 | 코드 | 주의점 |
|---|---|---|
| Import | `import gtsam` | Python 바인딩 기준 |
| Pose 키 생성 | `from gtsam.symbol_shorthand import X` → `X(0)` | `X(i)`는 변수 키 |
| 2D 포즈 | `gtsam.Pose2(x, y, theta)` | `theta`는 **rad** |
| 빈 그래프 | `gtsam.NonlinearFactorGraph()` | Factor들을 추가 |
| 상태값 저장 | `initial = gtsam.Values()` | 키마다 타입 유지 |
| 값 삽입 | `initial.insert(X(i), pose)` | 중복 키 삽입 주의 |
| 값 조회 | `result.atPose2(X(i))` | `x()`, `y()`, `theta()` |
| Prior | `gtsam.PriorFactorPose2(key, pose, noise)` | 절대 기준 |
| 상대포즈 제약 | `gtsam.BetweenFactorPose2(k1, k2, measurement, noise)` | **k1 좌표계 기준** 상대 변환 |
| 표준편차 | `gtsam.noiseModel.Diagonal.Sigmas(np.array([sx, sy, st]))` | 순서: x, y, θ |
| GN | `gtsam.GaussNewtonOptimizer(graph, initial).optimize()` | 초기값 민감할 수 있음 |
| LM | `gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()` | 감쇠(damping) 사용 |
| 전체 목적함수 | `graph.error(values)` | 작을수록 제약에 더 잘 부합 |
| 주변 공분산 | `gtsam.Marginals(graph, result).marginalCovariance(X(i))` | 국소 선형화 기반 |

### `Pose2`에서 상대포즈 읽기

```python
p0 = gtsam.Pose2(1.0, 2.0, 0.3)
p1 = gtsam.Pose2(2.0, 2.5, 0.4)
z01 = p0.between(p1)       # p0 좌표계에서 본 p0→p1 상대포즈
p1_reconstructed = p0.compose(z01)
assert np.allclose(
    [p1_reconstructed.x(), p1_reconstructed.y(), p1_reconstructed.theta()],
    [p1.x(), p1.y(), p1.theta()],
)
```

`BetweenFactorPose2(X(i), X(j), z_ij, ...)`에서 `z_ij`는 단순한 세계좌표 차분 `(x_j-x_i, y_j-y_i, θ_j-θ_i)`이 **아니다**. 회전이 포함된 **상대변환**이다.

## 2. 최적화 문제가 의미하는 것

각 포즈를 $T_i \in SE(2)$, 측정 상대변환을 $Z_{ij}$라 하자.

$$
T_i = \begin{bmatrix}R(\theta_i)&t_i\\0&1\end{bmatrix},\qquad
\hat Z_{ij}=T_i^{-1}T_j
$$

상대포즈 잔차는 개념적으로 다음과 같이 쓸 수 있다.

$$
r_{ij}(T_i,T_j) = \operatorname{Log}(Z_{ij}^{-1}T_i^{-1}T_j)
$$

> 위 식은 개념적인 Lie-group 표기다. GTSAM `Pose2` factor의 내부 오차 좌표계 및 차트 구현과 항목별 수식이 정확히 동일하다고 간주하면 안 된다.

가중 최소제곱의 목표는 다음과 같다.

$$
\min_{\mathbf T}\frac12\sum_k r_k(\mathbf T)^\top\Sigma_k^{-1}r_k(\mathbf T)
=\min_{\mathbf T}\frac12\sum_k\|r_k(\mathbf T)\|^2_{\Sigma_k^{-1}}.
$$

`Sigmas([σx, σy, σθ])`는 **분산이 아니라 표준편차**다. 독립 잡음 가정에서는 $\Sigma=\operatorname{diag}(\sigma_x^2,\sigma_y^2,\sigma_\theta^2)$이다.

- `sigma`가 **작을수록** 해당 측정을 강하게 신뢰한다.
- `sigma`가 **클수록** 해당 측정이 최적해에 미치는 영향은 작다.
- 측정 개수가 많다는 사실만으로 좋은 제약이 되는 것은 아니다. 노이즈 모델의 적절성이 중요하다.
- `graph.error(values)`는 통상 **$\frac12$이 포함된** factor error의 합이다. 사용하는 factor/noise model에 따라 해석하자.

### GN vs LM

선형화하여 $J\Delta\approx-r$를 풀 때,

$$
\text{GN:}\quad (J^\top WJ)\Delta=-J^\top Wr
$$

$$
\text{LM (개념적 형태):}\quad (J^\top WJ+\lambda D)\Delta=-J^\top Wr
$$

- **GN**: 초기값이 충분히 좋으면 빠른 편이다. 큰 업데이트가 실패를 유발할 수 있다.
- **LM**: 감쇠 파라미터로 업데이트 크기와 방향을 조절해 수렴 안정성을 높인다. `D`와 감쇠 규칙은 구현·설정에 따라 다르다.
- **최종 결과가 거의 같아도 정상**이다. 두 방법이 같은 목적함수의 같은 최소점에 수렴했을 수 있다. 차이는 반복 횟수, 초기 오차, 실패율에서 나타날 수 있다.

## 3. 최소 실행 예제: Prior + Odometry + Loop Closure

아래 코드는 **17개 포즈**가 사각형 경로를 따라 이동하는 예제다. (16번 이동, 시작점으로 복귀) 측정값에 잡음을 주고, 초기값은 noisy odometry 누적으로 생성한다. GN과 LM을 비교하고, **원본 상대측정값을 적분한 궤적(odometry dead reckoning)** 도 함께 그린다.

> `BetweenFactor`의 원본 관측값은 상대포즈이므로 세계좌표계의 점으로 바로 그릴 수 없다. 아래 `odom_xy`는 관측값을 순차 합성한 **시각화용 누적 궤적**이다.

```python
import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.symbol_shorthand import X

rng = np.random.default_rng(7)
N = 17

# 1) Ground truth: 네 변을 각 4 스텝씩 이동, 17개 포즈
steps = (
    [gtsam.Pose2(1.0, 0.0, 0.0)] * 4
    + [gtsam.Pose2(0.0, 1.0, 0.0)] * 4
    + [gtsam.Pose2(-1.0, 0.0, 0.0)] * 4
    + [gtsam.Pose2(0.0, -1.0, 0.0)] * 4
)
truth = [gtsam.Pose2(0.0, 0.0, 0.0)]
for step in steps:
    truth.append(truth[-1].compose(step))

# 2) 측정값/노이즈 정의
odom_sigmas = np.array([0.08, 0.08, 0.04])
loop_sigmas = np.array([0.05, 0.05, 0.03])
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
odom_noise = gtsam.noiseModel.Diagonal.Sigmas(odom_sigmas)
loop_noise = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)

graph = gtsam.NonlinearFactorGraph()
graph.add(gtsam.PriorFactorPose2(X(0), truth[0], prior_noise))

# 3) Noisy odometry measurements + dead reckoning
odom_measurements = []
dead_reckoning = [gtsam.Pose2(0.0, 0.0, 0.0)]
for i in range(N - 1):
    z_true = truth[i].between(truth[i + 1])
    eps = rng.normal(0, odom_sigmas)
    z_noisy = z_true.compose(gtsam.Pose2(*eps))
    odom_measurements.append(z_noisy)
    graph.add(gtsam.BetweenFactorPose2(X(i), X(i + 1), z_noisy, odom_noise))
    dead_reckoning.append(dead_reckoning[-1].compose(z_noisy))

# 마지막 포즈가 첫 포즈로 돌아와야 한다는 loop constraint
z_loop = truth[-1].between(truth[0])  # 이상적인 측정; 실제 SLAM에서는 noisy
# 독립 오차를 추가해 측정의 현실성을 높임
z_loop = z_loop.compose(gtsam.Pose2(*rng.normal(0, loop_sigmas)))
graph.add(gtsam.BetweenFactorPose2(X(N - 1), X(0), z_loop, loop_noise))

# 4) 초기값: dead reckoning을 바탕으로 만들기
initial = gtsam.Values()
for i, pose in enumerate(dead_reckoning):
    initial.insert(X(i), pose)

# 5) 최적화
initial_error = graph.error(initial)
gn_result = gtsam.GaussNewtonOptimizer(graph, initial).optimize()
lm_result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
print(f'initial error: {initial_error:.6f}')
print(f'GN error     : {graph.error(gn_result):.6f}')
print(f'LM error     : {graph.error(lm_result):.6f}')

# 6) 비교 시각화
def xy_from_poses(poses):
    return np.array([[p.x(), p.y()] for p in poses])

def xy_from_values(values, n=N):
    return np.array([
        [values.atPose2(X(i)).x(), values.atPose2(X(i)).y()]
        for i in range(n)
    ])

truth_xy = xy_from_poses(truth)
odom_xy = xy_from_poses(dead_reckoning)
gn_xy = xy_from_values(gn_result)
lm_xy = xy_from_values(lm_result)

plt.figure(figsize=(9, 8))
plt.plot(truth_xy[:, 0], truth_xy[:, 1], 'k:', label='Ground truth')
plt.plot(odom_xy[:, 0], odom_xy[:, 1], 'g.-', alpha=0.7,
         label='Noisy odometry (integrated) / Initial')
plt.plot(gn_xy[:, 0], gn_xy[:, 1], 'r--', label='Gauss-Newton')
plt.plot(lm_xy[:, 0], lm_xy[:, 1], 'b-', label='Levenberg-Marquardt')
plt.scatter(*truth_xy[0], c='black', marker='s', s=70, label='Start')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('GTSAM Pose2: GN vs LM vs Raw Odometry')
plt.show()
```

**해석**: GT는 예제에서만 알 수 있는 참값이다. `odom_xy`는 세계좌표계 원본 관측점이 아니라 odometry를 적분한 경로이며, 그 자체가 `initial`이다. GN·LM 궤적이 겹치는 경우 목적함수 값이나 반복 로그도 확인하자.

## 4. GN/LM의 초기값 민감도 실험

### 왜 나쁜 초기값을 넣어도 큰 차이가 안 날까?

1. 문제가 작고 loop closure가 강해서 최적점의 basin of attraction이 넓을 수 있다.
2. 초기값이 나빠 보여도 실제 상대포즈 잔차는 작을 수 있다. (전역 평행이동/회전은 prior의 영향이 핵심)
3. 두 optimizer 모두 같은 최소값으로 수렴하면 **최종 궤적 비교만으로는** 차이를 확인하기 어렵다.
4. 초기값을 `compose` 방식으로 왜곡하는지, 각 포즈의 세계좌표만 바꾸는지에 따라 비선형성이 달라진다.

### 효과적인 실험법

**A. 회전까지 크게 교란** — 초기값 생성 부분을 아래처럼 바꿔 재실행:

```python
bad_initial = gtsam.Values()
for i, p in enumerate(dead_reckoning):
    if i == 0:
        bad_initial.insert(X(i), p)
        continue
    dx, dy = rng.normal(0, 1.5, size=2)
    dtheta = rng.normal(0, 0.8)  # radians
    bad_initial.insert(
        X(i), p.compose(gtsam.Pose2(dx, dy, dtheta))
    )

print('bad initial error:', graph.error(bad_initial))
# 아래 최적화는 상황에 따라 실패할 수 있으므로 별도 실험으로 실행
# gn_bad = gtsam.GaussNewtonOptimizer(graph, bad_initial).optimize()
# lm_bad = gtsam.LevenbergMarquardtOptimizer(graph, bad_initial).optimize()
```

**B. 기록할 지표**: (i) 초기 error, (ii) 최종 error, (iii) 수렴 성공/실패, (iv) 반복 횟수·error 감소 추이, (v) 최종 위치 오차(참값이 있는 경우). `setVerbosity(...)` 등 파라미터 로그 기능의 정확한 선택지는 설치 버전의 `help(gtsam.GaussNewtonParams)`와 `help(gtsam.LevenbergMarquardtParams)`로 확인한다.

**C. 실험 설계**: 동일한 `graph`와 동일한 `bad_initial`을 두 optimizer에 전달한다. 교란 크기만 바꾸고 여러 random seed에서 **성공률과 최종 error 분포**를 비교한다. 일부 반복에서 GN 실패가 나오더라도 LM이 항상 성공한다고 일반화하면 안 된다.

## 5. 자주 발생하는 오류 및 디버깅

| 증상 | 의심할 원인 | 확인 방법 |
|---|---|---|
| 이상한 방향으로 궤적이 꺾임 | `BetweenFactor` 측정을 세계좌표 차분으로 사용 | `truth[i].between(truth[j])`로 비교 |
| θ가 비정상적으로 큼 | 도(degree)를 라디안으로 잘못 사용 | `np.deg2rad(...)` |
| 결과가 이동/회전해 떠다님 | Prior 누락 또는 약한 기준 | `PriorFactorPose2` 추가 및 σ 확인 |
| 최종 error가 커짐 | 틀린 loop closure, 노이즈 모델 부적절 | factor 하나씩 제거/추가하며 비교 |
| GN과 LM 결과가 동일 | 같은 해로 수렴 | 반복 수·에러 변화·실패율 비교 |
| key/type 조회 오류 | `X(i)` 누락, 해당 키가 없음, 타입 불일치 | `initial`/`result` 출력, 키 관리 |
| 그림에서 관측값이 안 보임 | 상대 측정값을 세계좌표 점으로 오인 | 연속 `compose`로 누적 궤적 생성 |

**필수 점검 코드**

```python
print('num factors:', graph.size())
print('initial error:', graph.error(initial))
print('optimized error:', graph.error(lm_result))
print('pose 0:', lm_result.atPose2(X(0)))
print('pose 16:', lm_result.atPose2(X(16)))

marginals = gtsam.Marginals(graph, lm_result)
print('covariance at X(16):\n', marginals.marginalCovariance(X(16)))
```

`marginalCovariance`는 결과 근방에서의 근사 불확실성을 나타내며, 실제 데이터의 편향·잘못된 데이터 연관까지 자동으로 반영하는 절대적 정확도 지표는 아니다.

## 6. Day 1 핵심 개념 체크

- [ ] `Pose2`의 `x, y, θ` 단위와 좌표계를 설명할 수 있다.
- [ ] `BetweenFactorPose2`의 측정은 **상대변환**임을 이해한다.
- [ ] `PriorFactorPose2`가 필요한 이유(gauge fixing)를 안다.
- [ ] `Sigmas`와 covariance/information의 차이를 설명한다.
- [ ] `graph`, `initial`, `optimizer`, `result`를 직접 연결할 수 있다.
- [ ] Odometry만 있는 경우와 loop closure 추가 시 결과를 비교한다.
- [ ] GN과 LM의 최종 결과가 같을 수 있는 이유를 이해한다.
- [ ] `graph.error(...)`와 궤적 시각화를 함께 확인한다.
- [ ] 원본 odometry가 **상대 측정**임을 구분하여 시각화한다.

## 7. Day 2로 이어지는 개념

| Day 1 (GTSAM / Python) | Day 2 확장 방향 |
|---|---|
| `Pose2`, `BetweenFactorPose2` | `Pose3`, SE(3), 3D SLAM |
| `NonlinearFactorGraph`, `Values` | g2o의 Vertex / Edge / Optimizer와 비교 |
| GN / LM | 선형 solver, sparse Jacobian/Hessian, robust kernel |
| Batch optimization | iSAM2 증분 최적화 |
| 단순 loop closure | 잘못된 loop / robust loss / switchable constraints |
| 수동 Factor 추가 | ROS에서 odom·sensor·loop closure를 그래프로 변환 |

## 참고 자료

- GTSAM 공식 시작하기: https://gtsam.org/get_started/
- GTSAM Factor Graph 개념: https://gtsam.org/tutorials/intro.html
- GTSAM Odometry Example: https://borglab.github.io/gtsam/odometryexample/
- GTSAM Pose2 SLAM Example: https://borglab.github.io/gtsam/pose2slamexample/

---

**한 줄 기억법**: **GTSAM = 그래프에 측정 제약을 넣고(`Factors`), 상태 초기값을 넣은 뒤(`Values`), 가중 비선형 최소제곱으로 일관된 상태를 찾는 도구.**
