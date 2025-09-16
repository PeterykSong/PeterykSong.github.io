---
#order: 70
layout: single
title: "Lecture 2-1 MVUE"
date: 2025-09-09 00:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900

excerpt: "Day 2"
tags:
  - EstimationTheory
  - Robotics
  
---

학교 수업 복습 칸입니다. 
{: .notice}

# 무엇을 추정(Estimation)하는 것인가?

공학을 하면서 수도없이 부딛히는 질문과 해답이 있다.   
측정한 데이터를 어떠한 모델로 바꾸고, 그 모델을 바탕으로 새로운 입력에 대한 출력을 예상하는 것.   
보통의 엔지니어들이 늘 요구받는 것들이다. 사실 사회학자들도 마찬가지겠다마는.....   

그래서, 어떤 입력(Input)이 물리 시스템(Physical System)을 거쳐 어떤 Output(데이터, 관측값, 측정값 등등등.)을 출력한다고 할때,   
이것을 Input / Model $\theta$ / x[n]  {x[0].x[1],......x[N-1]} 의 형태로 표현한다. 

우리는 여기서 모델을 나타내는 $\theta$ 를 어떻게 추정하는가가 이 과목의 핵심이다. 

## 1. Estimator의 정의

Estimator는 
- 표본(측정치, 관측치)을 입력으로 받아서, 
- 미지의 파라미터(확률적 모수)에 대한 **추정값(estimate)** 을 산출하는 
- **규칙(rule) 또는 함수(function)** 

$$ \text{Estimator : } \hat{\theta} = g(X_1,X_2,.....X_n)$$

>예: 모집단의 평균 μ를 알고 싶을 때, 표본의 평균 $\bar{X}$을 이용해 $\mu$를 추정한다면,  
>Estimator는 $\hat{\mu} = \frac{1}{n}\sum_{i=1}^n X_i$  
>Estimate는 실제 데이터로 계산된 값 (예: 5.2)

방금의 예시에서는 "평균"이라는 수단이 Estimator (이걸 추정량이라고 한다고..? ) 라고 할 수 있다. 

한 그룹에 키의 평균이 167인 집단이 있다. 이 집단의 누구를 뽑아도 그 사람은 키가 167에 비슷하다고 할 수 있다. 할수 있나? 여기서 두번째 질문을 확장해보자. 

## 2. Estimator 의 성질

Estimator 를 평가하는 몇가지 기준이 있다. 당연하게도. 

### 2.1. Bias
- 추정량(Estimator)의 기대값(평균)이 실제 모수와 같으면 불(不)편추정량(Unbiased estimator) :   $\mathbb{E}[\hat{\theta}] = \theta$
- 그렇지 않으면......... 편의(篇倚)추정량. 그래프가 옆으로 치우쳐졌다는 뜻. : $\mathbb{E}[\hat{\theta}] \neq \theta$

이때, $\mathbb{E}[\hat{\theta}] - \theta$ 를 편의 **bias** 라고 한다. 
$$ b(\theta) = \mathbb{E}[\hat{\theta}] - \theta$$

간단한 예제를 통해 전개를 한번 해보자. 
<br>
- $\hat{\theta}_1, \hat{\theta}_2, \dots, \hat{\theta}_n$ 를 $\theta$ 의 추정값"들"이라고 하자.(Estimations)

- 이때 새로운 Estimator 를 기존 estimator들의 평균으로 새롭게 정의해보자.  $  \hat{\theta} = \frac{1}{n} \sum_{i=1}^n \hat{\theta}_i.$ 

- 만약, 불편추정량이고(Unbiased), 상관관계가 없으며(Uncorrelated), 동일한 분산(with same variance)를 가진다면

  $$
  \mathbb{E}[\hat{\theta}] = \theta
  $$

  $$
  \mathrm{var}(\hat{\theta}) 
  = \frac{1}{n^2} \sum_{i=1}^n \mathrm{var}(\hat{\theta}_i) 
  = \frac{1}{n}\,\mathrm{var}(\hat{\theta}_1).
  $$

두번째 수식의 유도가 좀 궁금했다. 
1. 우선 분산의 성질을 알아보자. 
$$ \text{var}(aX) = a^2\text{var}(X)\,\,\,\,\,\ \text{상수는 제곱되어 튀어나온다.}$$  
$$ \text{var}(X+Y) = \text{var}(X) + \text{var}(Y)\,\,\,\,\,\ \text{둘이 상관관계가 없으면. Uncorrelated}$$  

2. 이걸 위의 식에 적용보자. 
$$
\mathrm{var}(\hat{\theta}) = \mathrm{var} \left(\frac{1}{n} \sum_{i=1}^n \hat{\theta}_i\right) = \frac{1}{n^2} \mathrm{var} \left(\sum_{i=1}^n \hat{\theta}_i\right)
$$

둘이 uncorrelated 면

$$
\mathrm{var} \left(\sum_{i=1}^n \hat{\theta}_i\right) = \mathrm{var} \left( \hat{\theta}_1+\hat{\theta}_2+\hat{\theta}_3......+\hat{\theta}_i\right) = \mathrm{var}(\hat{\theta}_1) +\mathrm{var}(\hat{\theta}_2) +\mathrm{var}(\hat{\theta}_3) ....\mathrm{var}(\hat{\theta}_i) = \sum_{i=1}^n \mathrm{var}(\hat{\theta}_i)
$$

3. 여기에.... 모든 $\hat{\theta}_i$가 같은 분산 $\sigma^2$를 가진다고 하면,
$$ \mathrm{var}(\hat{\theta}_i) = \frac{1}{n^2} \mathrm{var} \left(\sum_{i=1}^n \hat{\theta}_i\right)= \frac{1}{n^2}n\sigma^2 = \frac{\sigma^2}{n}  \tag{1}$$

이렇게 전개된 항에서, 만약 $ n \rightarrow \infty $ 가 되면, $ \mathrm{var} $ 는 0이 된다. 이는 분포가 점점 한 점에 수렴한다는 뜻으로, 의미론적으로 해석하면 $ \hat{\theta} = \theta $ 가 됨을 의미한다. 

그런데... 여기서 bias 가 있다면. 두둥. 

- 추정량 $\hat{\theta}_i$가 **biased estimator**라고 가정.
- 즉,
$$
\mathbb{E}[\hat{\theta}_i] = \theta + b(\theta)
$$
여기서 $b(\theta)$는 추정량의 **bias (편향)**.

- 평균 추정량 정의:
$$
\hat{\theta} = \frac{1}{n}\sum_{i=1}^n \hat{\theta}_i
$$
- 기대값은 선형성을 이용해:
$$
\mathbb{E}[\hat{\theta}] 
= \frac{1}{n}\sum_{i=1}^n \mathbb{E}[\hat{\theta}_i] 
= \frac{1}{n}\sum_{i=1}^n \big(\theta + b(\theta)\big)
= \theta + b(\theta)
$$
- **Bias는 평균을 내더라도 사라지지 않는다.**


$$
\mathbb{E} (\hat{\theta}) = \theta + b(\theta)  = \frac{1}{n} \sum_{i=1}^n \mathbb{E}(\hat{\theta}_i) 
$$

수식을 해석해보면, 
- 분산은 $n$이 커질수록 줄어들지만,
- bias가 그대로 남아 있기 때문에,
  $$
  \hat{\theta} \not\to \theta \quad (n \to \infty)
  $$
- 즉, **일치성(consistency)이 없다.**


### 2.2. Efficiency (효율성) 
같은 불편 추정기 중 분산이 가장 작은 경우

### 2.3 Consistency (일치성)
$n \to \infty$ 일 때 $\hat{\theta}_n \xrightarrow{p} \theta$

### 2. 4. Sufficiency (충분성)
데이터 속 파라미터에 대한 모든 정보를 담고 있음



## 3. Estimator의 종류

- 점 추정기 (Point Estimator): 하나의 값으로 추정
- 구간 추정기 (Interval Estimator): 신뢰구간으로 추정
- 선형 / 비선형 추정기
  
### 주요 예시

- 모평균: $\hat{\mu} = \frac{1}{n}\sum X_i$  
- 모분산: $\hat{\sigma}^2 = \frac{1}{n-1}\sum (X_i - \bar{X})^2$
- MLE: 데이터 중심, 가장 널리 쓰임
- MAP: 데이터 + 사전지식 결합
- MMSE: 평균제곱오차 기준 최적
- MVUE: 불편성과 최소분산 동시 만족


### 대표적인 Estimator 비교

| 추정 방법                                                      | 기본 아이디어                                 | 수학적 정의                                                                                                      | 장점                               | 단점                          |
| ---------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------- | --------------------------- |
| **MLE (Maximum Likelihood Estimator, 최대우도추정)**             | 관측된 데이터가 나올 확률을 최대화하는 파라미터 선택           | $\hat{\theta}*{MLE} = \arg\max*{\theta} L(\theta \mid X)$                                                 | - 직관적이고 일반적 <br>- 큰 표본에서 좋은 성질   | - 작은 표본에서 편향 가능 <br>- 계산 복잡 |
| **MAP (Maximum A Posteriori Estimator, 사후확률최대추정)**         | 우도 + 사전분포(Prior)를 고려해 사후확률이 최대인 파라미터 선택 | $\hat{\theta}*{MAP} = \arg\max*{\theta} p(\theta \mid X)$ <br>$ = \arg\max\_{\theta} p(X \mid \theta) p(\theta)$ | - Prior 반영 가능 <br>- 작은 표본에서도 안정적 | - Prior 선택이 주관적             |
| **MMSE (Minimum Mean Square Error Estimator, 최소평균제곱오차추정)** | 추정치와 참값의 평균 제곱 오차 최소화                   | $\hat{\theta}_{MMSE} = E[\theta \mid X]$                                                                | - 오차 최소화 기준 <br>- 확률적 불확실성 고려    | - 계산 복잡 <br>- 사후분포 필요       |
| **MVUE (Minimum Variance Unbiased Estimator, 최소분산 불편추정)**  | 불편추정기 중 분산이 최소인 추정기                     | $E[\hat{\theta}] = \theta $  & 최소 분산                                                                  | - 불편성과 효율성 동시 만족                 | - 항상 존재하지 않음 <br>- 계산 복잡    |



## 4. MSE
한줄 요약
$$
\mathrm{mse}(\hat{\theta}) = \mathbb{E}(\hat{\theta}-\theta)^2
$$
<br>
<br>
이걸 전개해보자.   

$$
\begin{align*}
\text{mse}(\hat{\theta}) 
&= \mathbb{E}\!\left[ \, (\hat{\theta} - \theta)^2 \, \right] \\[6pt]
&= \mathbb{E}\!\left[ \, \big(\hat{\theta} - \mathbb{E}[\hat{\theta}] + \mathbb{E}[\hat{\theta}] - \theta \big)^2 \, \right] \\[6pt]
&= \mathbb{E}\!\left[ \, \big( (\hat{\theta} - \mathbb{E}[\hat{\theta}]) + (\mathbb{E}[\hat{\theta}] - \theta) \big)^2 \, \right] \\[6pt]
&= \mathbb{E}\!\left[ (\hat{\theta} - \mathbb{E}[\hat{\theta}])^2 \right] 
   + \mathbb{E}\!\left[ (\mathbb{E}[\hat{\theta}] - \theta)^2 \right] \\[6pt]
&= \operatorname{Var}(\hat{\theta}) + \big(\mathbb{E}[\hat{\theta}] - \theta \big)^2 \\[6pt]
&= \operatorname{Var}(\hat{\theta}) + b(\theta)^2
\end{align*}
$$

즉 $ \mathrm{MSE} = \mathrm{Variance} + \mathrm{Bias}^2  $

수식이 잘 이해 안된다고? 이렇게 전개해보자. Thank you ChatGPT

$$
\text{MSE}(\hat{\theta}) = \mathbb{E}\big[(\hat{\theta} - \theta)^2\big]
$$

### (1) 기대값을 더하고 빼기
$$
\mathbb{E}\big[(\hat{\theta} - \theta)^2\big]
= \mathbb{E}\big[(\hat{\theta} - \mathbb{E}[\hat{\theta}] + \mathbb{E}[\hat{\theta}] - \theta )^2\big]
$$

여기서,
- $A = \hat{\theta} - \mathbb{E}[\hat{\theta}]$  
- $B = \mathbb{E}[\hat{\theta}] - \theta$

라 두면,  
$$
= \mathbb{E}[(A + B)^2]
$$



### (2) 제곱 전개
$$ 
\mathbb{E}[(A + B)^2] = \mathbb{E}[A^2 + 2AB + B^2]
$$



### (3) 기대값의 선형성과 항별 계산
$$
= \mathbb{E}[A^2] + 2\mathbb{E}[A]B + \mathbb{E}[B^2]
$$

- $\mathbb{E}[A] = \mathbb{E}[\hat{\theta} - \mathbb{E}[\hat{\theta}]] = 0$  
- $B$는 상수이므로 $\mathbb{E}[B^2] = B^2$

따라서,
$$
= \mathbb{E}[A^2] + B^2
$$


### (4) Var와 Bias로 표현
$$
= \operatorname{Var}(\hat{\theta}) + (\mathbb{E}[\hat{\theta}] - \theta)^2
$$

즉,
$$
\boxed{\text{MSE} = \text{Variance} + \text{Bias}^2}
$$


## 5. MVUE (Minimum Variance Un-biased Estimator)
- Find Estimator which minimizes the variance from a set of un-biased estimators.
- un-biased 추정량들 사이에서 최소 분산을 갖는 추정량을 찾는 과정이다. 

MVUE란 기댓값이 참값과 같은(불편성, unbiased) 추정량(Estimator)들 중에서, 그중 분산이 최소인 추정량(Estimator)를 찾는다 

- **불편 추정량 (Unbiased Estimator)**: $ \mathbb{E}[\hat{\theta}] = \theta $ 를 만족하는 추정기
- **MVUE**:   불편 추정량 중에서 **분산이 가장 작은 추정량**

$$ \hat{\theta}_{MVUE} = \arg\min_{\hat{\theta} \in U} \operatorname{Var}(\hat{\theta}) $$
($U$: Unbiased Estimator들의 집합)

### (예시 1). 
모집단: $X_1, X_2, \dots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$  
목표: $\mu$의 추정  

| 추정량 | 정의 | 불편성 확인 | 분산 |
|--------|------|-------------|------|
| 표본평균 $\hat{\mu}_1$ | $\hat{\mu}_1 = \frac{1}{n}\sum_{i=1}^n X_i$ | $ \mathbb{E}[\hat{\mu}_1] = \mu $ | $ \operatorname{Var}(\hat{\mu}_1) = \frac{\sigma^2}{n} $ |
| 첫 샘플 $\hat{\mu}_2$ | $\hat{\mu}_2 = X_1$ | $ \mathbb{E}[\hat{\mu}_2] = \mu $ | $ \operatorname{Var}(\hat{\mu}_2) = \sigma^2 $ |

첫 샘플의 $\sigma^2$는 모집단에서 이미 정의되었다. 
표본평균의 분산은 앞서 수식(1)을 참조하자. 저~~기 위에 있다. 

비교:  
$ \operatorname{Var}(\hat{\mu}_1) = \frac{\sigma^2}{n} \leq \sigma^2 = \operatorname{Var}(\hat{\mu}_2) $  
→ 따라서 **표본평균 $\hat{\mu}_1$** 이 MVUE.

### (예시 2): 모분산 $\sigma^2$ 추정

모집단: $X_1, \dots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$  

추정기:  
$ \hat{\sigma}^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2 $  

- 불편성: $ \mathbb{E}[\hat{\sigma}^2] = \sigma^2 $  
- 최소 분산도 만족  

→ 따라서 **$\hat{\sigma}^2$ 은 $\sigma^2$의 MVUE**.



## Appendix. Summary. Estimator / Estimate / 평균 추정량 구분

| 구분 | 기호 | 정의 | 성질 | 예시 |
|------|------|------|------|------|
| **모수 (Parameter)** | $\theta$ | 모집단의 참값 (고정, 미지) | 확률변수가 아님 | 모집단 평균 $\mu$ |
| **추정량 (Estimator)** | $\hat{\theta}$ | 표본 데이터의 함수 (랜덤) | 확률변수 | $\hat{\mu} = \frac{1}{n}\sum X_i$ |
| **추정값 (Estimate)** | $\hat{\theta}(x_1,\dots,x_n)$ | 실제 데이터 대입 후 계산된 값 | 고정된 수치 | $\hat{\mu} = 5.3$ |
| **여러 추정량** | $\hat{\theta}_1, \hat{\theta}_2, \dots, \hat{\theta}_n$ | 동일 모수를 다른 방식(혹은 독립 시도)으로 추정한 값들 | 서로 uncorrelated, unbiased 가정 | 각기 다른 실험에서 얻은 평균 추정량 |
| **평균 추정량 (Pooled Estimator)** | $\hat{\theta} = \frac{1}{n}\sum_{i=1}^n \hat{\theta}_i$ | 여러 추정량을 평균내어 만든 새로운 추정량 | 불편성 유지, 분산이 $1/n$로 감소 | 표본평균의 분산이 $\sigma^2/n$이 되는 원리와 동일 |

---


# 끝.