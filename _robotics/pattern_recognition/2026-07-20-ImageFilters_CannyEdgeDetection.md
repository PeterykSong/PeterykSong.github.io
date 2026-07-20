---
#order: 70
layout: single

title: "Image Filters - Canny Edge Detection"
date: 2026-07-20 20:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
related: false
topic : pattern_recognition
excerpt: "ImageFilter, CannyEdge"
tags:
  - Robotics
  - Pattern Recognition
  - Image Filters
  
---

논문을 읽으면서 나오는 이미지 필터/기하 추출방법들 실습칸입니다. 

{: .notice}

# 들어가면서.   

지난번에 리스트업을 하다보니 이미지 필터들이 무려 스무가지가 넘게 나왔다.   
최대한 빠른 시일안에 하나하나 밟아가보면서 공부를 좀 해보려 한다.  

챗지피티의 도움을 강력하게 받을 예정이므로.  
혹여 이 페이지 보고 공부하실 분들은 이 내용의 소스가 ChatGPT임을 미리 알아주셨으면 한다. 


# 1. Canny Edge Detection

앞서서 Gaussian Filter를 설명할때, 더 알아보는 과정에서 Edge 를 검출하는 Sobel 커널 이야기를  잠깐 했었다. 
이게 그거다. 

<figure>
  <img src="/assets/images/2026-07-20-21-57-48.png" style="width:80% !important; height:auto;" alt="2026-07-20-21-57-48">
  <figcaption>2026-07-20-21-57-48</figcaption>
</figure>


## 핵심 파라미터

```python
edges = cv2.Canny(
    blurred,
    threshold1=50,
    threshold2=150
)
```

Canny Edge Detection에서 핵심 파라미터는 `threshold1`과 `threshold2`이다.

- `threshold1`: 낮은 임계값
- `threshold2`: 높은 임계값

픽셀의 기울기 크기를 \(G(x,y)\), 낮은 임계값을 \(T_{\text{low}}\), 높은 임계값을 \(T_{\text{high}}\)라고 하자.

---

### `threshold1`: 낮은 임계값

`threshold1`은 약한 에지 후보를 구분하는 낮은 임계값이다.

기울기 크기 \(G(x,y)\)가 낮은 임계값 \(T_{\text{low}}\)보다 작으면 에지가 아닌 것으로 판단하여 제거한다.

$$
G(x,y) < T_{\text{low}}
$$

기울기 크기가 낮은 임계값 이상이면서 높은 임계값보다 작으면 약한 에지 후보로 분류한다.

$$
T_{\text{low}} \leq G(x,y) < T_{\text{high}}
$$

약한 에지는 주변의 강한 에지와 연결된 경우에만 최종 에지로 유지된다.

---

### `threshold2`: 높은 임계값

`threshold2`는 강한 에지를 구분하는 높은 임계값이다.

기울기 크기가 높은 임계값 \(T_{\text{high}}\) 이상이면 해당 픽셀을 강한 에지로 판단한다.

$$
G(x,y) \geq T_{\text{high}}
$$

강한 에지는 최종 결과에 직접 포함된다. 또한 주변에 연결된 약한 에지를 유지할지를 판단하는 기준으로 사용된다.

---

### 임계값에 따른 에지 분류

Canny Edge Detection의 에지 분류 규칙은 다음과 같이 정리할 수 있다.

$$
\text{Edge}(x,y)=
\begin{cases}
\text{비에지}, & G(x,y) < T_{\text{low}} \\[4pt]
\text{약한 에지}, & T_{\text{low}} \leq G(x,y) < T_{\text{high}} \\[4pt]
\text{강한 에지}, & G(x,y) \geq T_{\text{high}}
\end{cases}
$$

| 기울기 크기 | 분류 | 처리 |
|---|---|---|
| \(G(x,y) < T_{\text{low}}\) | 비에지 | 제거 |
| \(T_{\text{low}} \leq G(x,y) < T_{\text{high}}\) | 약한 에지 | 강한 에지와 연결된 경우에만 유지 |
| \(G(x,y) \geq T_{\text{high}}\) | 강한 에지 | 최종 에지로 유지 |

---

### 임계값 설정 예시

```python
edges = cv2.Canny(
    blurred,
    threshold1=50,
    threshold2=150
)
```

이 예시에서는 다음과 같이 동작한다.

#### 비에지

기울기 크기가 `50`보다 작으면 제거한다.

$$
G(x,y) < 50
$$

#### 약한 에지

기울기 크기가 `50` 이상이면서 `150`보다 작으면 약한 에지로 분류한다.

$$
50 \leq G(x,y) < 150
$$

#### 강한 에지

기울기 크기가 `150` 이상이면 강한 에지로 분류한다.

$$
G(x,y) \geq 150
$$

전체 분류를 하나의 식으로 표현하면 다음과 같다.

$$
\text{Edge}(x,y)=
\begin{cases}
\text{비에지}, & G(x,y) < 50 \\[4pt]
\text{약한 에지}, & 50 \leq G(x,y) < 150 \\[4pt]
\text{강한 에지}, & G(x,y) \geq 150
\end{cases}
$$

---

### 임계값의 일반적인 비율

일반적으로 높은 임계값은 낮은 임계값의 약 2배에서 3배 정도로 설정하는 경우가 많다.

$$
T_{\text{high}} \approx 2T_{\text{low}}
$$

또는

$$
T_{\text{high}} \approx 3T_{\text{low}}
$$

이를 범위로 표현하면 다음과 같다.

$$
2T_{\text{low}}
\leq
T_{\text{high}}
\leq
3T_{\text{low}}
$$

하지만 이는 절대적인 규칙이 아니다. 이미지의 밝기, 대비, 노이즈 수준과 검출하려는 구조의 특성에 따라 두 임계값을 조정해야 한다.


# 2. 실습 코드


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PeterykSong/FilterTutorial/blob/main/FilterTutorial/02_Canny_edge_detection.ipynb)

실습코드의 실행 결과는 다음과 같다. 

<figure>
  <img src="/assets/images/2026-07-20-22-00-35.png" style="width:80% !important; height:auto;" alt="2026-07-20-22-00-35">
  <figcaption>2026-07-20-22-00-35</figcaption>
</figure>


# 3. 알고리즘의 간단한 연산. Canny에서 Sobel 기울기 계산

Canny 내부에서는 일반적으로 Sobel 연산으로 수평·수직 기울기를 계산한다.

$$
G_x = I * S_x
$$

$$
G_y = I * S_y
$$

여기서 $I$는 입력 이미지이고, $*$는 컨볼루션 연산을 의미한다.

대표적인 Sobel 커널은 다음과 같다.

$$
S_x =
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$$

$$
S_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$

기울기 크기는 다음처럼 계산된다.

$$
G = \sqrt{G_x^2 + G_y^2}
$$

기울기 방향은 다음과 같다.

$$
\theta = \tan^{-1}\left(\frac{G_y}{G_x}\right)
$$

OpenCV에서는 `L2gradient=True`를 설정하면 위의 유클리드 거리 방식으로 기울기 크기를 계산한다.

```python
edges = cv2.Canny(
    blurred,
    threshold1=50,
    threshold2=150,
    apertureSize=3,
    L2gradient=True
)
```

`L2gradient=False`일 때는 더 빠른 근사식을 사용한다.

$$
G \approx |G_x| + |G_y|
$$

### 간단한 숫자 예시

어떤 픽셀에서 Sobel 연산 결과가 다음과 같다고 가정하자.

$$
G_x = 3
$$

$$
G_y = 4
$$

이때 기울기 크기는 다음과 같다.

$$
G = \sqrt{G_x^2 + G_y^2}
$$

$$
G = \sqrt{3^2 + 4^2}
$$

$$
G = \sqrt{9 + 16}
$$

$$
G = 5
$$

기울기 방향은 다음과 같다.

$$
\theta = \tan^{-1}\left(\frac{G_y}{G_x}\right)
$$

$$
\theta = \tan^{-1}\left(\frac{4}{3}\right)
$$

$$
\theta \approx 53.1^\circ
$$

따라서 이 픽셀의 기울기 크기는 약 $5$, 기울기 방향은 약 $53.1^\circ$이다.  
그러면 기울기의 비교 대상은 53.1도와 231.1도 방향에 있는 픽셀과 비교하게 되는데, 픽셀의 특징 완벽히 연속적인 값을 쓸순 없고 이산형으로 바꿔야 하니, 유사한 45도와 225도 방향의 픽셀과 비교하게 된다. 좀더 정밀한 구현을 위해 보간법을 쓰기도 한다고 한다.   

일단 여기까지만 알고 있자. 
