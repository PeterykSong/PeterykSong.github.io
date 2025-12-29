---
#order: 70
layout: single
classes: wide
title: "Lecture 2 PCA"
date: 2025-09-09 00:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900

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