---
#order: 70
layout: single

title: "Image Filters - Gaussian Filter"
date: 2026-06-15 00:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
related: false
topic : pattern_recognition
excerpt: "ImageFilter, GaussianFilter"
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


# 1. Gaussian Blur 

가우시안 블러. 이미지를 뿌옇게 만드는 가장 대표적인 방법이다. 
가장 기본이 되는 수식은 다음과 같다. 
중심점 주변의 픽셀을 불러와서 가우시안 값을 가중치로 곱해주는 것이다. 
중심점의 이동은 슬라이딩윈도우와 동일해서, 가중치의 곱셈(컨볼루션)이 끝나면 한픽셀 옆으로 이동한다. 
$$
B = I * G  \\

B(x,y) =\sum_{i=-k}^{k} \sum_{j=-k}^{k} G(i,j)\,I(x-i,y-j)
$$

$$
G(x,y)
=
\frac{1}{2\pi\sigma^2}
\exp\left(
-\frac{x^2}{2\sigma_x^2}  
-\frac{y^2}{2\sigma_y^2}
\right)
$$

대표적인 $3 \times 3$ Gaussian 근사 커널은 다음과 같다.

$$
K
=
\frac{1}{16}
\begin{bmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{bmatrix}
$$

이를 실제 가중치 값으로 나타내면 다음과 같다.

$$
K
=
\begin{bmatrix}
0.0625 & 0.125 & 0.0625 \\
0.125 & 0.25 & 0.125 \\
0.0625 & 0.125 & 0.0625
\end{bmatrix}
$$

중심 픽셀의 가중치는 $0.25$로 가장 크고, 모서리 픽셀의 가중치는
$0.0625$로 가장 작다.

커널의 모든 가중치 합은 다음과 같이 $1$이다.

$$
\sum_{i=-1}^{1}
\sum_{j=-1}^{1}
K(i,j)
=
1
$$

가우시안 함수 자체는 전체 영영게서 적분하면 1이 되어야 한다. 그러나 실제 적용할때는 픽셀을 유한하게 잘라서 샘플링하게 되므로 샘플링된 값이 정확히 1이 아닌경우가 발생한다. (그런가?) 그래서 다시 커널의 합을 $1$로 정규화 하는 과정을 거치는 편이 좋다. 이를 통해 이미지의 전체적인 값이 변하지 않도록 하기 위해서이다.

<figure>
  <img src="/assets/images/2026-07-20-17-03-17.png" style="width:80% !important; height:auto;" alt="2026-07-20-17-03-17">
  <figcaption>2026-07-20-17-03-17</figcaption>
</figure>

# 2. 실습

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PeterykSong/FilterTutorial/blob/main/FilterTutorial/01_GaussianBlur.ipynb)

# 3. 더 알아보기 

가우시안 블러는 가우시안 커널과 입력 이미지의 컨볼루션 연산이다.
즉, 가우시안 커널을 이용한 컨볼루션, 또는 가우시안 필터링이라고 말한다. 

그럼 다른 대표적인 커널들은 뭐가 있을까. 

### Edge Detection Kernel

Edge detection 필터는 이미지 밝기가 급격하게 변하는 지점을 검출한다.  
일반적으로 커널의 가중치 합은 $0$이며, 밝기가 일정한 영역에서는 출력값이 $0$에 가까워진다.

대표적인 수평 방향 밝기 변화 검출용 Sobel kernel은 다음과 같다.

$$
K_x
=
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$$

이 커널의 가중치 합은 다음과 같이 $0$이다.

$$
\sum_i \sum_j K_x(i,j)
=
-1+0+1-2+0+2-1+0+1
=
0
$$

입력 이미지의 밝기가 모두 일정한 값 $c$라고 하면, 출력값은 다음과 같다.

$$
\begin{aligned}
B(x,y)
&=
c\sum_i\sum_j K_x(i,j) \\
&=
c\cdot0 \\
&=
0
\end{aligned}
$$

따라서 밝기가 일정한 영역은 제거되고, 가로 방향으로 밝기가 변하는 경계가 강조된다.

수직 방향 밝기 변화를 검출하는 Sobel kernel은 다음과 같다.

$$
K_y
=
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$

두 방향의 결과를 각각 $G_x$와 $G_y$라고 하면, 최종 경계 강도는 다음과 같이 계산할 수 있다.

$$
G
=
\sqrt{
G_x^2+G_y^2
}
$$

계산량을 줄이기 위해 다음과 같은 근사식도 사용할 수 있다.

$$
G
\approx
|G_x|+|G_y|
$$

### Sharpening Kernel

Sharpening 필터는 중심 픽셀을 강조하고 주변 픽셀의 영향을 빼서 이미지의 경계와 세부 정보를 선명하게 만든다.

대표적인 sharpening kernel은 다음과 같다.

$$
K_{\mathrm{sharp}}
=
\begin{bmatrix}
0 & -1 & 0 \\
-1 & 5 & -1 \\
0 & -1 & 0
\end{bmatrix}
$$

이 커널의 가중치 합은 다음과 같이 $1$이다.

$$
\sum_i\sum_j K_{\mathrm{sharp}}(i,j)
=
0-1+0-1+5-1+0-1+0
=
1
$$

입력 이미지의 밝기가 모두 일정한 값 $c$인 경우, 출력은 다음과 같다.

$$
\begin{aligned}
B(x,y)
&=
c\sum_i\sum_j K_{\mathrm{sharp}}(i,j) \\
&=
c\cdot1 \\
&=
c
\end{aligned}
$$

따라서 일정한 밝기 영역은 대체로 유지되고, 밝기 변화가 있는 경계 부분은 강조된다.

이 sharpening kernel은 원본 이미지에서 Laplacian 성분을 빼는 형태로도 해석할 수 있다.

$$
I_{\mathrm{sharp}}
=
I-\nabla^2 I
$$

4-neighbor Laplacian kernel을 다음과 같이 정의하면,

$$
K_{\mathrm{lap}}
=
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

sharpening kernel은 다음 관계를 가진다.

$$
\begin{aligned}
K_{\mathrm{sharp}}
&=
K_{\mathrm{identity}}
-
K_{\mathrm{lap}} \\
&=
\begin{bmatrix}
0 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 0
\end{bmatrix}
-
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix} \\
&=
\begin{bmatrix}
0 & -1 & 0 \\
-1 & 5 & -1 \\
0 & -1 & 0
\end{bmatrix}
\end{aligned}
$$