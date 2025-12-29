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


<!-- PyScript CSS/JS 로드 -->
<link rel="stylesheet" href="https://pyscript.net/latest/pyscript.css" />
<script defer src="https://pyscript.net/latest/pyscript.js"></script>

<py-script>
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# 데이터 생성
mean0 = np.array([-1.0, 1.0])
cov0  = np.array([[0.4, 0.2],
                  [0.2, 0.3]])
X0 = np.random.multivariate_normal(mean0, cov0, size=100)

mean1 = np.array([2.0, -0.5])
cov1  = np.array([[0.3, -0.1],
                  [-0.1, 0.2]])
X1 = np.random.multivariate_normal(mean1, cov1, size=100)

X = np.vstack([X0, X1])

# PCA
mean = X.mean(axis=0)
X_centered = X - mean
cov = np.cov(X_centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
pc1 = eigvecs[:, 0]
pc2 = eigvecs[:, 1]

# LDA (2클래스)
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

# 시각화
fig, ax = plt.subplots(figsize=(5,5))

ax.scatter(X0[:,0], X0[:,1], alpha=0.5, label="class 0")
ax.scatter(X1[:,0], X1[:,1], alpha=0.5, label="class 1")

cx, cy = mean
scale = 3.0

# PCA
ax.arrow(cx, cy, scale*pc1[0], scale*pc1[1],
         width=0.03, color="red", length_includes_head=True)
ax.arrow(cx, cy, scale*pc2[0], scale*pc2[1],
         width=0.03, color="green", length_includes_head=True)

# LDA
cx2, cy2 = (m0 + m1) / 2.0
scale2 = 4.0
ax.arrow(cx2, cy2, scale2*w[0], scale2*w[1],
         width=0.03, color="blue", length_includes_head=True)

ax.set_aspect("equal", "box")
ax.set_title("PCA (red/green) & LDA (blue)")
ax.legend()

plt.show()
</py-script>