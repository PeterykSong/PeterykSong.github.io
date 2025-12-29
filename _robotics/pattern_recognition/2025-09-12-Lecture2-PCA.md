---
#order: 70
layout: single

title: "Lecture 2 PCA"
date: 2025-12-29 00:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
related: false

excerpt: "Day 2"
tags:
  - Robotics
  - Pattern Recognition
  - Machin Learning
  
---

학교 수업 복습 칸입니다. 
25년 Fall, Pattern Recognition and Machine Learning
{: .notice}

# 들어가면서.   

이 수업은 패턴인식론과 생성형 알고리즘을 다루는 수업이다. 
강의 순서는 다음과 같다. 

  1. Information
  2. PCA, LDA  --> 오늘
  3. Bayesian Decision
  4. Bayesian Network
  5. Parametric Estimation
  6. Non-parametric Estimation
  7. Variational Inference
  8. Deep Generative model

데이터분포를 어떻게 근사할 것인가. 그것이 핵심이다. 
지금은 데이터 전처리에서 가장 대표적으로 쓰인다. 

데이터있다고 가정해보자.  

{% raw %}
<!-- 1) PyScript CSS / JS 로드 -->
<link rel="stylesheet" href="https://pyscript.net/releases/2024.1.1/core.css">
<script defer type="module" src="https://pyscript.net/releases/2024.1.1/core.js"></script>

<!-- 2) 사용할 패키지 선언 -->
<py-config>
packages = ["numpy", "matplotlib"]
</py-config>

<!-- 3) PyScript 코드: PCA + LDA 2D 데모  -->
<py-script>
from pyscript import display
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# -------------------------
# 1. 데이터 생성
# -------------------------
# 클래스 0: 왼쪽 위쪽
mean0 = np.array([-1.0, 1.0])
cov0  = np.array([[0.4, 0.2],
                  [0.2, 0.3]])
X0 = np.random.multivariate_normal(mean0, cov0, size=100)

# 클래스 1: 오른쪽 아래쪽
mean1 = np.array([2.0, -0.5])
cov1  = np.array([[0.3, -0.1],
                  [-0.1, 0.2]])
X1 = np.random.multivariate_normal(mean1, cov1, size=100)

X = np.vstack([X0, X1])        # (N, 2)
y = np.array([0]*len(X0) + [1]*len(X1))

# -------------------------
# 2. PCA 계산
# -------------------------
mean = X.mean(axis=0)
X_centered = X - mean
cov = np.cov(X_centered.T)        # (2,2)

eigvals, eigvecs = np.linalg.eigh(cov)  # 작은 고유값부터
order = np.argsort(eigvals)[::-1]       # 내림차순 정렬
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

pc1 = eigvecs[:, 0]
pc2 = eigvecs[:, 1]

max_components = min(6, eigvals.shape[0])  # 최대 6개까지, 차원 수를 넘지 않게

lines = ["PCA Eigenvalues & Eigenvectors:"]
for i in range(max_components):
    vec = eigvecs[:, i]
    lines.append(
        f"PC{i+1}: eigenvalue = {eigvals[i]:.4f}, "
        f"vector = [{vec[0]:.4f}, {vec[1]:.4f}]"
    )

display("\n".join(lines))


# -------------------------
# 3. LDA 계산 (2 클래스 Fisher LDA)
# -------------------------
m0 = X0.mean(axis=0)
m1 = X1.mean(axis=0)

Sw = np.zeros((2,2))
for x in X0:
    d = (x - m0).reshape(2,1)
    Sw += d @ d.T
for x in X1:
    d = (x - m1).reshape(2,1)
    Sw += d @ d.T

w = np.linalg.inv(Sw) @ (m1 - m0).reshape(2,1)
w = w.ravel()
w = w / np.linalg.norm(w)

# -------------------------
# 4. 시각화
# -------------------------
fig, ax = plt.subplots(figsize=(5,5))

# 데이터 분포
ax.scatter(X0[:,0], X0[:,1], alpha=0.5, label="class 0")
ax.scatter(X1[:,0], X1[:,1], alpha=0.5, label="class 1")

# class 0 mean (m0)
ax.scatter(m0[0], m0[1],
           s=120, c="orange", marker="X",
           edgecolors="black", linewidths=1.2,
           label="mean m0")

ax.text(m0[0]+0.1, m0[1]+0.1,
        "m0", color="orange")

# class 1 mean (m1)
ax.scatter(m1[0], m1[1],
           s=120, c="purple", marker="X",
           edgecolors="black", linewidths=1.2,
           label="mean m1")

ax.text(m1[0]+0.1, m1[1]+0.1,
        "m1", color="purple")

# 전체 평균 (global mean)
ax.scatter(mean[0], mean[1],
           s=140, c="red", marker="D",
           edgecolors="black", linewidths=1.2,
           label="global mean")

ax.text(mean[0]+0.1, mean[1]+0.1,
        "global mean", color="red")
# PCA 화살표 (PC1, PC2)
cx, cy = mean
scale_pca = 3.0

ax.arrow(cx, cy, scale_pca*pc1[0], scale_pca*pc1[1],
         width=0.03, color="red", length_includes_head=True)
ax.text(cx + scale_pca*pc1[0]*1.1,
        cy + scale_pca*pc1[1]*1.1,
        "PC1", color="red")

ax.arrow(cx, cy, scale_pca*pc2[0], scale_pca*pc2[1],
         width=0.03, color="green", length_includes_head=True)
ax.text(cx + scale_pca*pc2[0]*1.1,
        cy + scale_pca*pc2[1]*1.1,
        "PC2", color="green")

# LDA 화살표
cx2, cy2 = (m0 + m1) / 2.0
scale_lda = 4.0

ax.arrow(cx2, cy2, scale_lda*w[0], scale_lda*w[1],
         width=0.03, color="blue", length_includes_head=True)
ax.text(cx2 + scale_lda*w[0]*1.1,
        cy2 + scale_lda*w[1]*1.1,
        "LDA", color="blue")

ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_aspect("equal", "box")
ax.set_title("PCA (red/green) & LDA (blue)")
ax.legend()

display(fig)
</py-script>
{% endraw %}

# PCA와 LDA의 목적

우선 위의 그래프는 아래의 코드를 통해 작성되었다. 

우선, 두개의 데이터군은 다음과 같다. 
Class0 은 -1,1을 평균으로 하여, 공분산값 cov0를 가지고, Class1 은 2,-0.5를 평균으로 공분산 cov1를 가진다. 

```python
# 클래스 0: 왼쪽 위쪽
mean0 = np.array([-1.0, 1.0])
cov0  = np.array([[0.4, 0.2],
                  [0.2, 0.3]])
X0 = np.random.multivariate_normal(mean0, cov0, size=100)

# 클래스 1: 오른쪽 아래쪽
mean1 = np.array([2.0, -0.5])
cov1  = np.array([[0.3, -0.1],
                  [-0.1, 0.2]])
X1 = np.random.multivariate_normal(mean1, cov1, size=100)
```

그럴때 PCA는 Class 0 + Class 1 이 가지는 전체 데이터 분포의 데이터 특성을 보여주고, LDA 는 Class0과 Class1과의 데이터를 구분하는 방향을 시각적으로 보여주고있다. 

이제 PCA와 LDA의 정의와 특성에 대해 곰곰히 뜯어보자. 

# PCA 

## PCA의 정의

> PCA는 고차원 데이터의 분산을 가장 잘 보존하는 방향을 찾아,       
> 저차원 공간으로 선형투영하는 차원 축소기법이다. 

우선, 앞서 그래프에서 PCA 벡터를 그리는 코드를 가져와보자. 


```python

mean = X.mean(axis=0)
X_centered = X - mean
cov = np.cov(X_centered.T)        # (2,2)

eigvals, eigvecs = np.linalg.eigh(cov)  # 작은 고유값부터
order = np.argsort(eigvals)[::-1]       # 내림차순 정렬
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

pc1 = eigvecs[:, 0]
pc2 = eigvecs[:, 1]

max_components = min(6, eigvals.shape[0])  # 최대 6개까지, 차원 수를 넘지 않게

lines = ["PCA Eigenvalues & Eigenvectors:"]
for i in range(max_components):
    vec = eigvecs[:, i]
    lines.append(
        f"PC{i+1}: eigenvalue = {eigvals[i]:.4f}, "
        f"vector = [{vec[0]:.4f}, {vec[1]:.4f}]"
    )

display("\n".join(lines))

```

코드에서 쉽게 보면, 데이터에서 PCA 분해는 Eigen-vector, 즉 고유값임을 알수 있다. -> eigvecs[:,order]
여기서 데이터는 2차원이므로 고유값이 2개이지만, 만약 차원이 늘어난다면 고유값은 차원의 개수많큼 존재한다. 

알아야 할 주요내용은 다음과 같다. 

1. 데이터 평균(mean)

$$
m = \frac{1}{n}\sum_{k=1}^{n} x_k
$$


2. 선형 투영 모델
데이터를 단위벡터 \(e\) 방향으로 투영

$$
x_k \approx m + a_k e
$$

투영 계수

$$
a_k = e^T(x_k - m)
$$

3. Scatter Matrix
산포 행렬

$$
S = \sum_{k=1}^{n} (x_k - m)(x_k - m)^T
$$
공분산 행렬과 같은 형태 (스케일만 다름)

4. PCA 최적화 문제 → 고유값 문제
분산을 최대로 하는 방향 \(e\) 를 찾는 문제

$$
\max_{e} \quad e^T S e
$$

이는 고유값 문제로 된다

$$
S e = \lambda e
$$

5. 주성분(Principal Components)
가장 큰 고유값에 대응하는 고유벡터가 1st PC

$$
e_1, e_2, \dots, e_d
$$

상위 \(d'\)개 축만 선택하여 투영

$$
a_i^{(k)} = e_i^T(x_k - m),
\qquad i = 1,\dots,d'
$$

6. 차원 축소 오차 (버린 정보량)
\(d'\)차까지 사용했을 때 남는(버려지는) 정보량

$$
J_{d'} =
\sum_{i=d'+1}^{d} \lambda_i
$$

## PCA의 특징

이렇게 해서 가지는 PCA의 결과(고유값)은 다음과 같은 특징들을 가진다. 

1) 비지도학습기반의 차원 축소 : 데이터 분포만 보고 축을 결정할 수 있다. 
2) 최대 분산의 보존 : 투영 후, 분산이 최대가 되도록 방향을 선택 -> 가장 많은 정보가 담긴 축을 찾는다. 
3) 직교축의 생성 : 고유벡터들간에는 서로 직교한다. 
4) 최소제곱오차를 가진다.(MSE)  : 원본데이터를 투영선으로 근사했을때, 재구성 오차가 최소다. 
5) 낮은 고유값은 버려지는 축이다 : 작은 고유값은 기여도가 작은 방향, 차원 축소시 우선 제거된다. 

## 왜 PCA?

고차원의 데이터는 
  - 중복된 정보가 많고  
  - 축 간 상관성이 존재하는 경우가 많으며
  - 계산비용과 저장 비용이 커지는 문제가 있다. 

따라서, **정보는 최대한 유지하면서 차원을 줄이는 방법** 이 된다. 
차원을 몇까지 줄일 것인가는 데이터의 특성을 보고, 잘 판단해야 하는 문제가 남아있다마는. 그래도 한차원 줄일때마다 연산량을 극적으로 낮출수 있다는 장점을 눈여겨봐야 한다. 

Scatter Matrix를 보면, {(데이터) - (평균)} 을 제곱한 것들의 합인 것을 알수 있다.   
죽, PCA는 오차의 제곱이 작아지는 방향을 찾는 문제로 설명할 수 있으며, 
이 수식을 전개해보면, 결국 고유값을 찾는 문제로 귀결될 수 있음을 알 수 있다. 
(유도할 줄 알면 좋다. 근데, 유도식을 어디에서 쓰긴 할까....???)

그렇게 얻은 고유값을 각각의 축으로 하여 데이터를 투영시키면 그것을 이 데이터의 **"주성분(Principle Components)"**이라고 이야기 한다. 

## PCA가 잘될때와 안될때

1. 잘 되는 경우
  - 데이터가 타원형 분포일때
  - 축 사이의 강한 상관관계가 있을때
  - 노이즈의 방향이 작은 고유값에 몰릴때 (노이즈를 날리면 되니까)
  - 저차원구조가 선형적으로 존재할때

 이런 경우 상위 몇개 축만으로도 대부분의 분산(variance)를 설명가능하다.

2. 잘 안되는 경우
  - 클래스 정보가 중요하지만 분산이 작을때
  - 테이터 구조가 비선형 다양체(Manifold)
  - 이상치(Outlier)가 크고 분산을 강하게 왜곡할때
  - 스케일이 다른 feature가 섞여있을때
  이런 경우 전처리(정규화, 표준화)를 통해 다시 PCA 해석을 수행해볼 수 있겠다. 

## 
