---
#order: 70
layout: single
title: "Lecture 1-1 LTI"
date: 2025-09-09 00:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900

excerpt: "Day 1"
tags:
  - EstimationTheory
  - Robotics
  
---

학교 수업 복습 칸입니다. 
{: .notice}


## 사족

수업들으면서 공부하는 내용을 적기 위해 블로그 구조를 조금 바꿨다가 3시간동안 헤맸다. 
내가 완전히 파악한 블로그 코드가 아닌데, 무리하게 너무 많은 것을 하려 하지 않았나 하는 자책을 한다. 
어쨌거나 포맷을 만들었으니 이제 적극적으로 써보자. 


## 오늘의 총평

학부생도 수강이 가능한 수업이어서, 첫날은 쉽게 쉽게 간다. 
수학적인 기초를 한번 훑어 지나가보자. 


# 1. Matrix

행렬 연산은 늘 볼때마다 헷갈린다. 주요 연산 공식을 다시 정리해보자. 

$$ 
\begin{aligned}
(AB)^T &= B^TA^T\\
 AA^{-1} &= A^{-1}A = I\\
 AB^{-1} &= B^{-1}A^{-1}\\
 (A^T)^{-1} &= (A^{-1})^T\\
 Tr(AB) &= Tr(BA)\\
 Tr(ABC) &= Tr(CAB) = Tr(BCA)\\
 |AB| &= |A||B|\\
 |A^{-1}| &= \frac{1}{|A|}    ... 이건 좀 생소하다.\\
   \end{aligned}
 $$

늘상 Eignevector/Eigne value 관련해서는 익숙하지가 않다.   
이 부분은 별도로 더 공부해서 보완하도록 하자. (나중에 링크 걸어둘 것.)

<div style="text-align: center;">
  <img src="/assets/images/robotics/estimation_lec1_1.png" alt="고유값 특성">
  <figcaption>Eigen Vector/Value에 대하여 <br> <br> </figcaption>
</div>

<br>
<br>
<br>

# 2. Linear system

현재 강의에서는 선형시스템을 기본으로 한다. 
기본적으로 선형시스템은 아래와 같이 정의한다. 

### 2-1. 선형 시스템의 일반 형태
$$
\dot{x}(t) = A x(t) + B u(t), \qquad y(t) = C x(t) + D u(t)
$$

- $A$: 상태천이 행렬 (system dynamics)  
- $B$: 입력 행렬  
- $C$: 출력 행렬  
- $D$: feedthrough (직접 전달항)

---

그러나 교재에서는 아래처럼 표현하고 있다. 

$$\begin{aligned}
\dot{x} &= Ax + Bu \\
  y &= Cx 
\end{aligned}$$

어라. D는 없네. 좀 더 찾아보니 이런 의미가 있다고 한다. 맞나? 


> ## D = 0
> 입력이 출력으로 즉시 전달되지 않는다.   
> $y(t) = Cx(t) + D u(t)$ 인데, $D$ 가 0이므로, Proper system 또는 Strictly proper system이라고 한다.   
> + Proper : 전달함수의 분자가 분모보다 차수가 크지 않은 경우 
> + Strictly proper system : 전달함수의 분자가 분모보다 반드시 낮은 찾수일때. 

이때 이 시스템의 해는 다음과 같다. 

$$
\begin{aligned}
x(t) &= e^{A(t-t_0)}x(t_0)+ \int^t_{t0} e^{A(t-\tau)}Bu(\tau)d\tau \\
y(t) &= Cx(t)
\end{aligned}
$$

state transition matrix : $$ e^{At} = \sum_{k=0}^{\infty} \frac{(At)^k}{k!}$$
강의 노트에는 테일러 급수라고 살짝 메모가 되어있는데, 정확히 말하면 행렬 지수(matrix exponential)를 정의하는 급수, 또는 멱급수(Power series)라고 말한다. 

아래는 챗지피티의 도움을 받아 작성한 내용이다. 

---

### 2-2. 상태천이행렬 (State Transition Matrix)

$$
e^{At} = \sum_{k=0}^{\infty} \frac{(At)^k}{k!}
$$

- **해석**: 상태방정식 $\dot{x}(t)=Ax(t)$의 해는
  $$
  x(t)=e^{At}x(0)
  $$

- **성질**
  - $\frac{d}{dt}e^{At}=Ae^{At}=e^{At}A$  
  - $e^{A\cdot 0}=I$  
  - $e^{A(t+s)}=e^{At}e^{As}$  

---

### 2-3. 입력 포함 해
$$
x(t)=e^{At}x(0) + \int_0^t e^{A(t-\tau)}Bu(\tau)\,d\tau
$$

출력은
$$
y(t)=C e^{At}x(0) + C\int_0^t e^{A(t-\tau)}Bu(\tau)\,d\tau + D u(t)
$$

- $D=0$ → 출력은 반드시 상태를 통해서만 형성됨  
- $D \neq 0$ → 출력에 즉시 반응 포함  


---

### 2-4. 전달함수 관점
$$
G(s)=C(sI-A)^{-1}B+D
$$


- $D=0$ → strictly proper (분모 차수 > 분자 차수)  
- $D\neq 0$ → 상수항 존재 (feedthrough)  

---

### 2-5. 2×2 예제
$$
A=\begin{bmatrix}-1&0\\0&-2\end{bmatrix},\quad
B=\begin{bmatrix}1\\1\end{bmatrix},\quad
C=\begin{bmatrix}1&1\end{bmatrix},\quad
u(t)=1
$$

- 상태:
  $$
  x(t)=
  \begin{bmatrix}1-e^{-t}\\[2pt]\tfrac{1-e^{-2t}}{2}\end{bmatrix}
  $$

- 출력:
  - $D=0$: $\; y(t)=\tfrac{3}{2}-e^{-t}-\tfrac{1}{2}e^{-2t}, \;\; y(0)=0$  
  - $D=2$: $\; y(t)=\tfrac{7}{2}-e^{-t}-\tfrac{1}{2}e^{-2t}, \;\; y(0^+)=2$


### 2-6. 행렬 지수의 테일러 전개  

$$
e^{At} = \sum_{k=0}^{\infty} \frac{(At)^k}{k!}
$$

- 이는 스칼라 지수 함수 $e^z$의 **테일러 급수**를 행렬에 확장한 것  
- 따라서 “테일러 급수 전개”라고 불러도 되고, 보다 엄밀히는 **행렬 지수의 멱급수 정의**라고 함

(참고) 스칼라 지수함수
$$
e^{z} = \sum_{k=0}^{\infty} \frac{z^k}{k!} \qquad z \in \mathbb{C}
$$


### 관련 과목
이 내용은 보통 다음 수업에서 다룸:
- **신호 및 시스템 (Signals and Systems)**  
- **선형제어 / 현대제어 (Linear Control, Modern Control Theory)**  
- **선형시스템 이론 (Linear Systems)**  
- **선형대수학 + 미분방정식** (기초)    

<br>
<br>
<br>

# 3. Stability

+ 모든 유한한 초기상태 $x(0)$ 에 대해, $x(t)$ 가 모든 t에 대해 발산하지 않는다면(유계하다면/bounded), Marginally stable 하다고 할 수 있다. 
+ Marginally stable if $x(t)$ is bounded for all t and for and for all bounded initial state x(0)
+ 모든 유한한 초기상태 $x(0)$에 대하여 $ \displaystyle \lim_{t \to \infty} x(t) = 0$ 일때 점근적으로 안정적이라고 말한다. (Asymptotically stable)

이하 Theorem

### 3-1안정성 정리 (Theorems)
---

- **Marginally stable**:  
  A continuous-time linear time invariant (LTI) system is marginally stable  
  if and only if  
  
  $$
  \lim_{t \to \infty} \|e^{At}\| \leq M < \infty
  $$
  for some matrix $M$.  
  
  연속시간 LTI 시스템은 상태천이행렬이 발산하지 않고 어떤 유한한 상수 $M$ 이하로 항상 억제될 때 한계적으로 안정하다.
  → 해가 발산하지 않고 유계 상태에 머무는 경우.

---

- **Asymptotically stable**:  
  A continuous-time LTI system is asymptotically stable  
  if and only if  
  $$
  \lim_{t \to \infty} e^{At} = 0.
  $$
  시스템이 안정하려면 → 시간이 지남에 따라 반드시 원점으로 수렴.

---

- **Eigenvalue condition (marginal stability)**:  
  A continuous-time LTI system is marginally stable if and only if  
  - all eigenvalues of $A$ have negative real parts, **or**  
  - all eigenvalues of $A$ have negative or zero real parts, and  
    those with zero real parts have **geometric multiplicity = algebraic multiplicity**.  
  
  - 모든 고유값이 음수 실수부를 가지거나,
  - 일부 고유값이 실수부가 0일 수 있지만(허수축 위에 있음), 이때는 기하학적 중복도(geometric multiplicity)가 대수적 중복도(algebraic multiplicity)와 같아야 함.
(즉, Jordan block이 크기가 1이어야 함 → 발산 없이 단순 진동만 유지)

  → 허수축 위 고유값이 있어도 Jordan block이 크기 1이면 단순 진동만 존재.

---

- **Eigenvalue condition (asymptotic stability)**:  
  A continuous-time LTI system is asymptotically stable  
  if and only if all eigenvalues of $A$ have **negative real parts**.  
  모든 고유값의 실수부가 음수일 때 : $t \to \infty$에서 0으로 수렴.

<br>
<br>
<br>

# 4. Controllability
선형 시스템 $ \dot{x} = Ax+Bu$  에서, 어떠한 초기값 $x(0)$ 과 어떠한 최종상태 $x_1$ 에 대해 유한한 시간 $t$ 에서 $x(0)$ 를 $x_1$으로 천이(transfer)시켜줄수 있을때, (A,B) 는 제어 가능하다고 말할 수 있다. 

지피티 says.

쌍 $(A,B)$가 제어가능(controllable) 하다고 말하는 조건:
임의의 초기 상태 $x(0)$와 임의의 최종 상태 $x_1$에 대해,
어떤 입력 $u(t)$를 가하여 유한한 시간 안에 $x(0)$에서 $x_1$로 상태를 옮길 수 있으면, 시스템은 제어가능하다.

+ 시스템이 controllable 하다는 건, 입력 $u(t)$를 잘 설계하면 상태 공간(state space)의 모든 점을 도달할 수 있다는 뜻
+ 즉, 시스템을 원하는 상태로 "조종(control)"할 수 있는 성질.
+ 반대로 controllable 하지 않다면, 아무리 입력을 바꿔도 도달할 수 없는 상태들이 존재한다는 말. 


### 4-1. Controllability: Equivalent Conditions/ 동치조건 (Continuous-Time LTI)

Consider
$$
\dot{x}=Ax+Bu,\qquad x\in\mathbb{R}^n,\;u\in\mathbb{R}^m.
$$

- **(A,B) is controllable.**
  - By definition: for any $x(0)$ and any $x_1$, there exists an input $u(t)$ that transfers $x(0)$ to $x_1$ in finite time.

---

- **Controllability matrix has full row rank.**
  $$
  \mathcal{C}=\big[\,B\;\;AB\;\;A^2B\;\cdots\;A^{n-1}B\,\big]\in\mathbb{R}^{n\times nm},
  \qquad \operatorname{rank}(\mathcal{C})=n.
  $$

---

- **Controllability Gramian (finite-horizon) is nonsingular.**
  $$
  W_c(t)=\int_{0}^{t} e^{A\tau}BB^{\top}e^{A^{\top}\tau}\,d\tau,
  \quad t>0.
  $$
  If $W_c(t)\succ0$ (nonsingular/positive definite) for some $t>0$, then $(A,B)$ is controllable.
  Conversely, if $(A,B)$ is controllable, then $W_c(t)\succ0$ for all $t>0$.

---

- **PBH test (Popov–Belevitch–Hautus).**
  For every eigenvalue $\lambda$ of $A$,
  $$
  \operatorname{rank}\,[A-\lambda I\;\;B]=n.  \to \text{full rank}
  $$

---

- **Lyapunov equation (when $A$ is Hurwitz).**
  If all eigenvalues of $A$ have negative real parts, then the unique solution $W_c$ of
  $$
  A W_c + W_c A^{\top} = -BB^{\top}
  $$
  is positive definite. (Equivalently, $W_c=\int_{0}^{\infty} e^{A\tau}BB^{\top}e^{A^{\top}\tau}d\tau \succ 0$.)



교재에 생략된 내용이 많다.. 1시간도 안되는 수업시간동안 한학기 내용을 축약했구나 싶다. 


<br>
<br>
<br>


# 5. Observability

### 5-1. 가관측성 (Observability)의 정의
---

연속시간 선형시스템
$$
\dot{x}(t) = A x(t) + B u(t), \quad y(t) = C x(t) + D u(t)
$$

- 쌍 $(A,C)$가 **가관측 가능(observable)** 하다고 한다는 것은:

  - 임의의 초기 상태 $x(0)$와  
  - 임의의 유한 시간 $t > 0$에 대해,  
  - 입력 $u(s)$와 출력 $y(s)$ ($0 \leq s \leq t$)를 모두 알고 있다면,  

  초기 상태 $x(0)$를 **유일하게 결정할 수 있다**는 것을 의미한다.



#### 직관적인 의미
- **가관측성**: 출력과 입력 신호를 통해 시스템 내부 상태를 완전히 복원할 수 있음  
- **제어가능성**: 입력을 통해 시스템 상태를 원하는 대로 만들 수 있음  

따라서,  
- 제어가능성과 가관측성은 서로 **쌍대(dual)** 관계에 있다.


### 5-2. 가관측성 (Observability) 판별 조건
--- 

연속시간 선형시스템:
$$
\dot{x}(t) = A x(t) + B u(t), \quad y(t) = C x(t) + D u(t)
$$

쌍 $(A,C)$가 **가관측 가능(observable)** 하다는 것은, 출력과 입력 정보를 통해 초기 상태 $x(0)$를 유일하게 복원할 수 있다는 뜻이다.

---

#### 5-2-1. 관측가능성 행렬 (Observability matrix)
$$
\mathcal{O} =
\begin{bmatrix}
C \\
CA \\
CA^2 \\
\vdots \\
CA^{n-1}
\end{bmatrix}
\in \mathbb{R}^{n \times n}
$$
- $\operatorname{rank}(\mathcal{O}) = n$ 이면 $(A,C)$는 가관측 가능.

---

#### 5-2-2. 관측가능성 그라미안 (Observability Gramian)
$$
W_o(t) = \int_0^t e^{A^\top \tau} C^\top C \, e^{A \tau}\, d\tau
$$
- $W_o(t)$가 $t>0$에서 **양정정부호(nonsingular)** 이면 가관측 가능.

---

#### 5-2-3. PBH 테스트 (Popov–Belevitch–Hautus test)
모든 $A$의 고유값 $\lambda$에 대해,
$$
\operatorname{rank}
\begin{bmatrix}
A - \lambda I \\
C
\end{bmatrix} = n
$$
이면 $(A,C)$는 가관측 가능.

---

#### 5-2-4. 리아푸노프 방정식 (Lyapunov equation)
$A$가 Hurwitz 행렬(모든 고유값의 실수부 < 0)일 경우,
$$
A^\top W_o + W_o A = -C^\top C
$$
의 해 $W_o$가 **양정정부호**이면 $(A,C)$는 가관측 가능.

---

### 요약
- **제어가능성**: 입력으로 상태를 원하는 곳으로 이동시킬 수 있음  
- **가관측성**: 출력으로부터 내부 상태를 알아낼 수 있음  
- 두 성질은 서로 **쌍대(dual)** 관계에 있다.

<br>
<br>
<br>

# 6. Stabilizability and Detectability (안정화 가능성과 검출 가능성)


- **원문:**  
  If a system is **controllable or stable**, then it is also *stabilizable*.  
  If a system is **uncontrollable or unstable**, then it is *stabilizable* if its uncontrollable modes are **stable**.  

- **해석:**  
  어떤 시스템이 **제어가능(controllable)** 하거나 **안정(stable)** 하다면,  
  그 시스템은 항상 **안정화 가능(stabilizable)** 하다.  

  만약 시스템이 **제어불가능(uncontrollable)** 하거나 **불안정(unstable)** 하더라도,  
  제어할 수 없는 모드(uncontrollable modes)가 **안정**하다면,  
  그 시스템은 여전히 **안정화 가능(stabilizable)** 하다.

---

- **원문:**  
  If a system is **observable or stable**, then it is also *detectable*.  
  If a system is **unobservable or unstable**, then it is *detectable* if its unobservable modes are stable.  

- **해석:**  
  어떤 시스템이 **가관측 가능(observable)** 하거나 **안정(stable)** 하다면,  
  그 시스템은 항상 **검출 가능(detectable)** 하다.  

  만약 시스템이 **비가관측(unobservable)** 이거나 **불안정(unstable)** 하더라도,  
  관측할 수 없는 모드(unobservable modes)가 **안정**하다면,  
  그 시스템은 여전히 **검출 가능(detectable)** 하다.   

<br>
<br>
     

| 구분 | 강한 조건 | 약한 조건 | 설명 |
|------|-----------|-----------|------|
| 제어가능성 (Controllability) | 모든 상태를 입력으로 제어할 수 있음 | – | 가장 강한 성질 |
| 안정화가능성 (Stabilizability) | – | 제어 불가능한 모드가 존재해도 **그 모드들이 안정**이라면 허용 | 제어가능성보다 약한 조건 |
| 가관측성 <br> (Observability) | 출력으로 모든 상태를 추정할 수 있음 | – | 가장 강한 성질 |
| 검출가능성 (Detectability) | – | 관측 불가능한 모드가 존재해도 **그 모드들이 안정**이라면 허용 | 가관측성보다 약한 조건 |

---

### 요약
- **제어가능성 ⇒ 안정화가능성** (항상 성립)  
- **가관측성 ⇒ 검출가능성** (항상 성립)  
- 반대 방향은 일반적으로 성립하지 않음.  
- 따라서, 안정화가능성과 검출가능성은 각각 제어가능성과 가관측성의 **완화된 조건**이라고 볼 수 있다.


<br>
<br>
<br>

# Appendix. 시스템의 모드 (Modes)

## 1. 모드의 정의
- 모드(mode)란 선형 시스템에서 **행렬 $A$의 고유값(eigenvalue)과 대응하는 고유벡터 방향**으로 표현되는 **기본 동특성**을 의미한다.
- 즉, 시스템 해를 구성하는 **기본 지수항 $e^{\lambda t}$**에 해당한다.

---

## 2. 상태해에서의 모드
연속시간 LTI 시스템:
$$
\dot{x} = A x
$$

해는
$$
x(t) = e^{At} x(0).
$$

만약 $A$가 대각화 가능하다면,
$$
x(t) = \sum_i c_i e^{\lambda_i t} v_i
$$

- $\lambda_i$: 고유값 → 모드의 시간 거동 (감쇠, 발산, 진동)
- $v_i$: 고유벡터 → 상태 공간에서의 방향
- $c_i$: 초기 조건에 따른 계수

따라서 각 항 $e^{\lambda_i t} v_i$가 시스템의 **모드**가 된다.

---

## 3. 안정성과 모드
- $\operatorname{Re}(\lambda_i) < 0$: 모드는 시간이 지남에 따라 **감쇠** (안정)
- $\operatorname{Re}(\lambda_i) > 0$: 모드는 시간이 지남에 따라 **발산** (불안정)
- $\operatorname{Re}(\lambda_i) = 0$: 모드는 **진동** 또는 상수 유지 (한계 안정)

---

## 4. 제어가능성과 모드
- 입력 행렬 $B$에 의해 특정 모드를 제어할 수 있으면 → **제어가능 모드(controllable mode)**
- 제어 불가능한 모드가 있더라도 그것이 안정 모드라면 → 시스템은 **stabilizable**

---

## 5. 관측가능성과 모드
- 출력 행렬 $C$에 의해 특정 모드를 관측할 수 있으면 → **관측가능 모드(observable mode)**
- 관측 불가능한 모드가 있더라도 그것이 안정 모드라면 → 시스템은 **detectable**

---

## 2차 시스템 예제

다음 시스템을 고려하자:
$$
A = \begin{bmatrix}0 & 1 \\ -2 & -3 \end{bmatrix}
$$

### 1) 고유값 계산
특성방정식:
$$
\det(A - \lambda I) = \lambda^2 + 3\lambda + 2 = 0
$$
따라서
$$
\lambda_1 = -1, \quad \lambda_2 = -2
$$

→ 두 모드 모두 **실수부가 음수**이므로 시스템은 **점근 안정(asymptotically stable)**.

---

### 2) 고유벡터 계산
- $\lambda_1 = -1$에 대한 고유벡터:
$$
(A + I)v = 0 \;\;\Rightarrow\;\; v_1 = \begin{bmatrix}1 \\ -1\end{bmatrix}
$$

- $\lambda_2 = -2$에 대한 고유벡터:
$$
(A + 2I)v = 0 \;\;\Rightarrow\;\; v_2 = \begin{bmatrix}1 \\ -2\end{bmatrix}
$$

---

### 3) 해 표현
따라서 상태해는   

$$
x(t) = c_1 e^{-t} \begin{bmatrix}1 \\ -1\end{bmatrix}
+ c_2 e^{-2t} \begin{bmatrix}1 \\ -2\end{bmatrix},
$$   

여기서 $c_1, c_2$는 초기 조건 $x(0)$에 의해 결정된다.

---

### 4) 해석
- 모드 1: $e^{-t}$ → 느리게 감쇠하는 성분  
- 모드 2: $e^{-2t}$ → 더 빠르게 감쇠하는 성분  
- 따라서 전체 시스템은 두 감쇠 모드의 조합으로 움직이며, 시간이 지나면 $x(t)\to 0$으로 수렴한다.


<br>
<br>
<br>

## 제어 불가능 모드 예시

### 예제 1) 제어 불가능 + 안정화 불가 (λ=0 모드가 제어 불가능)

시스템
$$
A=\begin{bmatrix}0&0\\0&-1\end{bmatrix},\quad
B=\begin{bmatrix}0\\1\end{bmatrix}.
$$

#### 1) 제어가능성 행렬(Controllability matrix)
$$
\mathcal{C}=[\,B\;\;AB\,]
=\begin{bmatrix}
0 & 0\\
1 & -1
\end{bmatrix},\qquad
\operatorname{rank}(\mathcal{C})=1<2.
$$
⇒ **제어 불가능**.

#### 2) PBH 테스트
- 고유값: $\lambda_1=0,\;\lambda_2=-1$

$\lambda=0$에 대해
$$
[A-\lambda I\;\;B]
=
\begin{bmatrix}
0& 0& 0\\
0&-1& 1
\end{bmatrix},
\quad \operatorname{rank}=1<2
\Rightarrow \text{($\lambda=0$ 모드 제어 불가능)}.
$$

$\lambda=-1$에 대해
$$
[A-\lambda I\;\;B]
=
[A+I\;\;B]
=
\begin{bmatrix}
1&0&0\\
0&0&1
\end{bmatrix},
\quad \operatorname{rank}=2
\Rightarrow \text{($\lambda=-1$ 모드 제어 가능)}.
$$

#### 3) 해석
상태방정식은
$$
\dot{x}_1=0,\qquad \dot{x}_2=-x_2+u.
$$
- $x_1$은 **입력과 무관**하게 상수 유지($\lambda=0$ 모드) → **제어 불가능 모드**.  
- $x_2$는 입력으로 제어 가능($\lambda=-1$ 모드).

### 4) 안정화 가능성 (Stabilizability)
- 제어 불가능 모드가 **$\lambda=0$ (한계 안정)** → 일반적으로 안정화 가능성은 **Re($\lambda$)<0**(엄격 안정)만 허용.  
- 따라서 이 시스템은 **stabilizable이 아님**.

---

### 예제 2) 제어 불가능 + 안정화 가능 (제어 불가능 모드가 엄격 안정)

시스템
$$
A=\begin{bmatrix}-2&0\\0&-1\end{bmatrix},\quad
B=\begin{bmatrix}0\\1\end{bmatrix}.
$$

#### 1) 제어가능성 행렬
$$
\mathcal{C}=[\,B\;\;AB\,]
=\begin{bmatrix}
0 & 0\\
1 & -1
\end{bmatrix},\qquad
\operatorname{rank}(\mathcal{C})=1<2.
$$
⇒ **제어 불가능**.

#### 2) PBH 테스트
- 고유값: $\lambda_1=-2,\;\lambda_2=-1$

$\lambda=-2$에 대해
$$
[A-\lambda I\;\;B] =[A+2I\;\;B] =
\begin{bmatrix}
0&0&0\\
0&1&1
\end{bmatrix},
\quad \operatorname{rank}=1<2
\Rightarrow \text{($\lambda=-2$ 모드 제어 불가능)}.
$$

$\lambda=-1$에 대해
$$
[A-\lambda I\;\;B]
=
[A+I\;\;B]
=
\begin{bmatrix}
-1&0&0\\
0&0&1
\end{bmatrix},
\quad \operatorname{rank}=2
\Rightarrow \text{($\lambda=-1$ 모드 제어 가능)}.
$$

#### 3) 해석
$$
\dot{x}_1=-2x_1,\qquad \dot{x}_2=-x_2+u.
$$
- $x_1$ ($\lambda=-2$)는 **입력으로 건드릴 수 없지만 자체적으로 지수 감쇠** → 제어 불가능 **이지만 안정 모드**.  
- $x_2$ ($\lambda=-1$)는 입력으로 제어 가능.

### 4) 안정화 가능성
- 제어 불가능 모드의 고유값 $-2$가 **Re($\lambda$)<0** 이므로,  
  이 시스템은 **stabilizable** (상태 피드백으로 전체를 안정하게 만들 수 있음).

---

### 핵심 요약
- **제어 불가능 모드**: 어떤 고유값 방향이 입력 $B$로는 전혀 자극될 수 없는 경우  
  - 판별: $\operatorname{rank}[A-\lambda I\;\;B]<n$ (PBH), 또는 $\operatorname{rank}(\mathcal{C})<n$.  
- **Stabilizability**: 제어 불가능 모드가 존재해도 **그 모드들이 모두 Re($\lambda$)<0**이면 OK.  
  - 예제 1: $\lambda=0$ → **stabilizable 아님**  
  - 예제 2: $\lambda=-2$ → **stabilizable**


<br>
<br>
<br>
<br>

20년이 지나서야 대략 이해가 되는구나........