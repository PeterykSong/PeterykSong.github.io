---
#order: 70
layout: single

title: "Noise for Pattern, Diffusion. Day1"
date: 2026-10-07 18:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
related: false
topic : pattern_recognition
excerpt: "Diffusion"
tags:
  - Robotics
  - Pattern_Recognition
  - Machine_Learning
  - Semantic Segmentation
  - Diffusion
  
---

# 들어가면서. 

지난번까지는 UNet 을 이용해서 패턴을 인식하는 과정을 한번 거쳐봤다. 문제는 특정한 형태의 도면이 아니면 패턴의 인식 확율이 떨어지는 문제가 있다는 것도 확인했다. 그러면 이제 그 형태를 어떻게 핸들링할 것인가 하는 질문을 해 볼 수 있다. 

그래서 생각해볼 수 있는건 생성모델을 통해 캐드형태의 도면이 아닌 것을 캐드형태로 바꾼 다음, 거기서 다시 원하는 Feature를 추출하는 방식으로 접근할 수 있지 않을까 생각하게 된 것이다. 

생성형 모델에는 Diffusion과 GAN이 가장 대표적이다. VAE도 클래식이긴 한데, 앞서 두가지에 비해 최근 애용되지는 않는듯하다. 여튼 그 중 Diffusion을 먼저 실습해보면서, 이후에 GAN도 같이 확인해보자. 그러면 이 둘의 차이를 알수 있지 않을까 싶기도 하다. 

# 학습 순서

Day 1의 목표는 Diffusion Model의 이미 생성 원리를 이해하고, 간단하게 모델을 직접 실행해보는것까지 하도록 한다. 

따라서, 아래 질문들을 머릿속에 염두해두면서 학습을 진행하도록 하자. 

1. Diffusion Model은 어떻게 이미지를 생성하는가?
2. 이미지에 노이즈를 추가하는 이유는 무엇인가?
3. 노이즈를 제거하는 모델은 어떻게 학습하는가?
4. DDPM은 어떤 구조로 동작하는가?
5. 건축 약도를 도면으로 변환할 때 Diffusion을 어떻게 활용할 수 있는가?


# Forward Diffusion 이해하기. 

Diffusion Model은 크게 두가지로 구성된다. 하나는 Foward Diffusion이라고 해서, 원본이미지에 조금씩 노이즈를 추가하는 과정을 말한다. 그리고 반대로 노이즈가 있는 이미지에서 단계적으로 노이즈를 제거하는 과정을 Reverse Diffusion이라고 한다. 

- $x_0$ 원본이미지
- $x_t$ 노이즈가 추가된 이미지
- $x_T$ 원본을 못알아볼 정도로의 노이즈

1. Foward Diffusion 
$$
x_0 \to x_1 \to x_2 \to ..... \to x_T
$$

2. Reverse Diffusion 

$$
x_T \to x_{T-1} \to x_{T-2} \to ..... \to x_0
$$

이 일련의 과정들을 Neural Network, 즉 신경망이 학습하고 연산하게 만드는 과정을 말한다. 

<figure>
  <img src="/assets/images/2026-10-19-13-52-54.png" style="width:80% !important; height:auto;" alt="2026-10-19-13-52-54">
  <figcaption>https://test.dagshub.com/blog/how-to-evaluate-generative-image-models/</figcaption>
</figure>


수식을 유도하는 과정은 혁펜하임같은 유명강의를 참조하면 좋다.  
다만, 여기선 그냥 실습을 통해 빠르게 따라가보자.  

# 실습 1. 몸풀기

일단 간단한 흑백이미지를 하나 그려보자. 

```python
import numpy as np
import matplotlib.pyplot as plt

# 64x64 흰색 이미지 생성
image = np.ones((64, 64))

# 벽 구조 생성
image[10:54, 10:13] = 0
image[10:54, 51:54] = 0
image[10:13, 10:54] = 0
image[51:54, 10:54] = 0

# 내부 벽
image[10:40, 30:33] = 0

plt.figure(figsize=(5, 5))
plt.imshow(image, cmap="gray", vmin=0, vmax=1)
plt.title("Original Floor Plan")
plt.axis("off")
plt.show()
```
그러면 단순한 흑/백 평면도같은게 하나 보인다. 
<figure>
  <img src="/assets/images/2026-10-19-22-27-03.png" style="width:80% !important; height:auto;" alt="2026-10-19-22-27-03">
  <figcaption>2026-10-19-22-27-03</figcaption>
</figure>

## Forward Process / Gaussian Noise 추가

Diffusion에서 가장 중요한 요소는 Gaussian Noise다. 

$$
\epsilon \sim \mathcal{N}(0,I)
$$

평균은 0이고, 분산이 1인 (Identity matrix) 정규분포에서 노이즈를 생성한다는 의미다. 

```python
noise = np.random.randn(64, 64) # Random함수를 이용해 noise를 생성했다. 


noisy_image = image + 0.3 * noise  # 그리고 이 noise를 이미지 행렬에 더했다. 계수도 한번 바꿔보자. 

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=1)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap="gray", vmin=0, vmax=1)
plt.title("Noisy Image")
plt.axis("off")

plt.show()
```
<figure>
  <img src="/assets/images/2026-10-19-22-30-47.png" style="width:80% !important; height:auto;" alt="2026-10-19-22-30-47">
  <figcaption>2026-10-19-22-30-47</figcaption>
</figure>

노이즈가 추가 되면 이런 형태가 된다. 주석에서 볼수 있듯, 0.3이라는 계수를 바꿔보면 노이즈가 더 심해짐을 알 수 있다. 
이걸 몇번에 걸쳐서 반복하면 이미지가 점점 노이즈화되는걸 알 수 있다. 이제 다음으로 넘어가자. 

# 실습 2. DDPM Forward Diffusion 구현

Diffusion의 대표 알고리즘인 DDPM을 해보자. 
DDPM은 단순히 이미지에 노이즈를 더하는 것이 아니라, 정해진 확률적 규칙에 따라 이미지를 노이즈로 변환한다  

이번 실습에선 아래 네가지를 이해하면 된다. 

1. DDPM에서 노이즈를 추가하는 수학적 원리
2. $\beta_t$, $\alpha_t$, $\bar{\alpha}_t$ 의 의미
3. PyTorch를 이용한 Forward Diffusion 구현

## 이론
앞서서 Difusion, 특히 Forward diffusion 에서는 원본 이미지에 Gaussian Noise를 조금씩 추가해나가는 과정이라 말한 바 있다. 

$$
x_0 \to x_1 \to x_2 \to ..... \to x_T
$$

이때, 각 단계는 다음의 확률분포를 다룬다. 

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t ; \sqrt{1-\beta_t x_{t-1}},\beta_t, I)
$$

| 기호 | 의미 |
|---|---|
| $x_0$ | 원본 이미지 |
| $x_t$ | 시간 $t$에서의 이미지 |
| $\beta_t$ | 해당 단계에서 추가되는 노이즈의 분산 |
| $T$ | 전체 Diffusion 단계 수 |
| $I$ | 단위행렬 |

$\beta_t$ 는 매 단계마다 노이즈를 얼마나 추가할건지에 결정한다. 일반적으로 초기에는 작은 노이즈를 추가하고, 시간이 갈수록 더 많은 노이즈를 추가한다. 이러한 설정을 **Noise Schedule** 이라고 한다. 

그러면 앞서 알아야 한다는 $\alpha_t$ 는 뭐인가. 말로 표현해본다면, 현재단계에서 이전 신호가 유지되는 비율과 관련된 값이다. 

$$
\alpha_t = 1 - \beta_t
$$

$\bar{\alpha}_t$ 는 지금단계까지의 $\alpha_t$ 값의 누적곱이 되며, 원본의 이미지가 얼마나 유지되는가를 뜻한다.

$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

이 $\bar{\alpha}_t$ 를 이용하면 DDPM에서 중간단계의 이미지를 직접 생성할 수 있다. 

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
$$

위 수식에서 $\sqrt{\bar{\alpha}_t} x_0$ 는 원본이미지성분에 해당하고, $\sqrt{1-\bar{\alpha}_t} \epsilon$는 노이즈 성분에 해당하게 된다. 

## Noise Schedule 구현

이제 코드로 구현하면서 먼저 셋팅부터 하고, 이제 평면도를 가정한 이미지 하나를 넣어 tensor로 변환하자. 


```python
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# 이미지 크기
H, W = 64, 64

# 흰색 배경
image = np.ones((H, W), dtype=np.float32)

# 외벽
image[10:54, 10:13] = 0
image[10:54, 51:54] = 0
image[10:13, 10:54] = 0
image[51:54, 10:54] = 0

# 내부 벽
image[10:40, 30:33] = 0

# PyTorch Tensor 변환
x0 = torch.tensor(
    image,
    dtype=torch.float32,
    device=device
)

# DDPM에서 일반적으로 사용하는 [-1, 1] 범위로 변환
x0 = x0 * 2 - 1

print("Shape:", x0.shape)
print("Device:", x0.device)
print("Min:", x0.min().item())
print("Max:", x0.max().item())
```

중간의 주석에서 [-1,1] 의 범위로 변환한다고 써저있다. 이는 이미지 생성모댈에선 입력값을 일정한 범위로 정규화(Normalize)한다. 여기서는 다음과 같이 변환하게 된다. 

| 원본 픽셀 | 변환된 픽셀 |
|---|---|
| 검은색 0 | -1 |
| 회색 0.5 | 0 |
| 흰색 1 | 1 |

이제 본격적으로 Noise scheule을 생성해보자. DDPM에서 사용하는 가장 기본적인 방법은 Linear Noise Schedule이다. 전체 단계를 1000으로 설정해보자. (1000번 노이즈 넣는다는 이야기다.)


```python
T = 1000

beta_start = 0.0001
beta_end = 0.02

betas = torch.linspace(
    beta_start,
    beta_end,
    T,
    device=device
)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(
    alphas,
    dim=0
)

print("Betas shape:", betas.shape)
print("Alpha bars shape:", alpha_bars.shape)
```
여태 보지 못한 함수가 하나 등장했다. `torch.cumprod()` 인데, 이는 앞서 이론 설명할때 말했든 누적곱을 뜻한다. 

지금까지의 $\beta$,$\alpha$값들을 시각화 해보면 다음과 같다. 

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(betas.cpu().numpy())
plt.title("Beta Schedule")
plt.xlabel("Time Step")
plt.ylabel("Beta")

plt.subplot(1, 3, 2)
plt.plot(alphas.cpu().numpy())
plt.title("Alpha Schedule")
plt.xlabel("Time Step")
plt.ylabel("Alpha")

plt.subplot(1, 3, 3)
plt.plot(alpha_bars.cpu().numpy())
plt.title("Cumulative Alpha")
plt.xlabel("Time Step")
plt.ylabel("Alpha Bar")

plt.tight_layout()
plt.show()
```

<figure>
  <img src="/assets/images/2026-10-19-23-50-24.png" style="width:80% !important; height:auto;" alt="2026-10-19-23-50-24">
  <figcaption>2026-10-19-23-50-24</figcaption>
</figure>

위를 다시 해석해보면, 
① Beta Schedule : 시간이 증가하면서 노이즈 분산이 증가한다.
② Alpha Schedule : 시간이 증가하면서 $\alpha_t$ 가 감소한다.
③ Cumulative Alpha : $\bar{\alpha}_t$ 가 점차 0에 가까워진다. 이게 핵심이다. t가 증가할수록 원본이미지는 거의 남아있지 않게 된다는 뜻이다. 

## Forward Diffusion 함수 구현
이제 핵심함수를 구현해보자. 

```python
def forward_diffusion(x0, t, alpha_bars):

    # 누적 alpha 값
    alpha_bar_t = alpha_bars[t]

    # Gaussian Noise 생성
    noise = torch.randn_like(x0)

    # Forward Diffusion
    xt = (
        torch.sqrt(alpha_bar_t) * x0
        +
        torch.sqrt(1 - alpha_bar_t) * noise
    )

    return xt, noise
```

코드 자체는 매우 심플 단순하다.

Forward Diffusion이 구현된 줄에서 아래 식이 잘 구현되어있는지 한번 눈여겨보자. 

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
$$


## 변화 확인

이제 t 가 진행되면서 이미지가 어떻게 변화하는지 한번 보도록 하자. 

```python
timesteps = [0, 50, 100, 200, 400, 600, 800, 999]

plt.figure(figsize=(16, 8))

for i, t in enumerate(timesteps):

    xt, noise = forward_diffusion(
        x0,
        t,
        alpha_bars
    )

    xt_img = xt.cpu().numpy()

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        xt_img,
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    plt.title(f"t = {t}")
    plt.axis("off")

plt.tight_layout()
plt.show()
```

<figure>
  <img src="/assets/images/2026-10-19-23-54-21.png" style="width:80% !important; height:auto;" alt="2026-10-19-23-54-21">
  <figcaption>2026-10-19-23-54-21</figcaption>
</figure>

단계값인 t가 증가하면서 점점 노이즈가 강해지고 ,500을 점어가는 순간부터는 거의 원본이미지를 알아볼수가 없을 정도가 된다. 앞서서 그래프에서도 설명되었지만, Alpha와 Beta는 반비례관계를 가지고 있다. 

## 추가 실험 — Noise Schedule 변경

기존 설정은 beta값을 다음과 같이 설정해서 변화를 주고 있다. 

```python
beta_start = 0.0001
beta_end = 0.02
```
이 값을 바꿔보자. 

- beta_end = 0.005
<figure>
  <img src="/assets/images/2026-10-20-00-30-55.png" style="width:80% !important; height:auto;" alt="2026-10-20-00-30-55">
  <figcaption>2026-10-20-00-30-55</figcaption>
</figure>

- beta_end = 0.05
<figure>
  <img src="/assets/images/2026-10-20-00-31-25.png" style="width:80% !important; height:auto;" alt="2026-10-20-00-31-25">
  <figcaption>2026-10-20-00-31-25</figcaption>
</figure>

beta 값의 변화를 어떻게 설정하는가에 따라 노이즈의 정도가 달라진다. 노이즈가 천천히 추가된다고 해서 반드시 생성 성능이 좋아지는 것은 아닙니다. 적절한 스케줄은 학습 방식과 데이터에 따라 달라진다. 해봐야 할게 많단 소리다. 


# 실습 3. DDPM Reverse Diffusion 구현

그래. 노이즈를 입히는 과정은 그렇다 치자. 그냥 더하는 과정이다 라고 하자. (확율의 제곱근에 원본이미지와 노이즈를 곱해서 더했다. 수식참조) 그러면 이 노이즈에서 원본 이미지를 어떻게 다시 구해내는건가. 그 말에 답을 하기 위해서 하나더 해봐야 할게 있다. CNN이 노이즈를 예측하도록 학습해야 한다는 것이다. 

앞서 구현한 식을 다시 가져와보자. 


$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
$$

저 식을 통해 작성한 Forward Diffusion에서는 우리가 직접 노이즈를 생성했으므로 $\epsilon$ 을 알고 있다. 그렇다면, 신경망이 다음 문제를 풀수 있게 학습만 시키면 된다. 

>노이즈가 포함된 이미지 $x_t$와 시간 $t$가 주어졌을 때, 원래 추가된 노이즈 $\epsilon$을 예측하라.

그러면 위의 수식에서 schedule과 노이즈를 알게 되었으니 $x_0$가 나오는게 아닌가?  이것이 DDPM의 핵심이다. 

## Noise Prediction Network 

노이즈를 예측하는 신경망을 다음과 같이 표현한다. 

$$
\epsilon_\theta (x_t,t)
$$

수식에서 볼 수 있듯, 이 식은 두가지 정보를 입력받는다. 하나는 노이즈가 포함된 이미지 $x_t$이고, 현재 Diffusion 단계인 시간 t다. 이 두개를 입력받아 노이즈값을 CNN모델을 통해 예측하고, 이걸 학습하는 과정에서는 MSE Loss를 사용한다. 

## DDPM의 손실함수

DDPM의 대표적인 Objective function은 다음과 같다. 

$$
L_simple = \mathcal{E}_{x_0,t,\epsilon} \left[ \Vert \epsilon - \epsilon_\theta ( x_t,t)\Vert ^2 \right]
$$

수식은 복잡해보이지만, 의미는 간단하다. 실제로 추가한 노이즈와 신경망이 예측한 노이즈를 최대한 비슷하게 만드는 것이다.   

가령 실제 노이즈가 $\epsilon=[0.2,-0.5,0.8]$ 이고, 예측 노이즈가 $\epsilon_\theta=[0.1,-0.3,0.7]$ 라면 MSE는 다음과 같이 게산된다. 

$$
MSE=\frac{(0.2-0.1)^2+(-0.5+0.3)^2+(0.8-0.7)^2}{3}
$$

직접 계산해보자. 값은 0.02다. 

이걸 python으로 구현하면 다음과 같다. 

```python
import torch
import torch.nn.functional as F

actual_noise = torch.tensor(
    [0.2, -0.5, 0.8],
    device=device
)

predicted_noise = torch.tensor(
    [0.1, -0.3, 0.7],
    device=device
)


#Loss함수의 정의. 한줄로 쉽게 된다. 
loss = F.mse_loss(predicted_noise,actual_noise)

print("MSE Loss:", loss.item())
```


## 학습데이터의 생성

앞서서는 이미지 하나만 가지고 실습을 했지만, 노이즈를 신경망을 통해 "학습"이란걸 해야 하기에, 여러이미지의 데이터셋이 필요하다. 아직은 기초 실습이니 random함수를 통해 자동으로 생성해보도록 하자. 

- 이미지 크기: 64×64
- 흰색 배경
- 검은색 외벽
- 무작위 내부 벽
- 학습 이미지: 2,000개

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_floorplan(size=64):

    img = np.ones((size, size), dtype=np.float32)

    # 외벽
    img[8:56, 8:11] = 0
    img[8:56, 53:56] = 0
    img[8:11, 8:56] = 0
    img[53:56, 8:56] = 0

    # 내부 벽 개수
    n_walls = np.random.randint(1, 5)

    for _ in range(n_walls):

        x = np.random.randint(14, 48)
        y = np.random.randint(14, 48)

        length = np.random.randint(10, 30)

        if np.random.rand() > 0.5:
            # 수직 벽
            img[y:min(y + length, 53), x:x+2] = 0

        else:
            # 수평 벽
            img[y:y+2, x:min(x + length, 53)] = 0

    return img
```
위의 함수를 이용해서 실제 데이터셋을 만들자. 

```python
N = 2000

floorplans = np.stack([
    generate_floorplan()
    for _ in range(N)
])

print("Dataset shape:", floorplans.shape)
```

만들어진 데이터를 한번 훑어볼까. 

```python
plt.figure(figsize=(12, 6))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        floorplans[i],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.axis("off")

plt.tight_layout()
plt.show()
```

<figure>
  <img src="/assets/images/2026-10-20-01-21-37.png" style="width:80% !important; height:auto;" alt="2026-10-20-01-21-37">
  <figcaption>2026-10-20-01-21-37</figcaption>
</figure>

이 데이터를 학습하려면 Tensor로 바꿔야 한다. 

```python
from torch.utils.data import TensorDataset, DataLoader

# [0,1] → [-1,1]
floorplans = floorplans * 2 - 1

# (N,H,W) → (N,C,H,W)
floorplans = torch.tensor(
    floorplans,
    dtype=torch.float32
).unsqueeze(1)

dataset = TensorDataset(floorplans)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

print("Tensor shape:", floorplans.shape)
print("Number of batches:", len(dataloader))
```
여기까지 실행해보면, 다음과 같이 출력된다. 

<figure>
  <img src="/assets/images/2026-10-20-01-43-58.png" style="width:80% !important; height:auto;" alt="2026-10-20-01-43-58">
  <figcaption>2026-10-20-01-43-58</figcaption>
</figure>

개, 채, 행, 렬이니까. 2000개의 이미지. 채널은 1개. 이미지사이즈는 64x64인걸 확인할 수 있다. 



## Time Embedding
앞서서 Diffusion을 설명하면서, 시간 t에 따라 노이즈의 양이 달라진다고 이야기한 바있다. 이걸 반영해야 한다. 방금의 이미지 데이터셋 텐서에 시간정보를 붙여야 한다.   


그러나, 우리가 Tensor를 다루면서 간과한게 있다면 tensor는 integer가 아니란거다. (뭐.. 있긴하겠지만.)

따라서, 시간에 대한 정보를 정규화해서 붙여넣는다. 이걸 embedding 한다고 한다. 


$$
t_normalized = \frac{t}{T-1}
$$

이렇게 하면 전체 시간(혹은 단계)에서 지금의 텐서가 어디쯤에 위치하는지 0~1사이로 정규화 할수 있다. 채널로 만들어 붙여 넣자. 그러기 위해선 이미지와 같은 크기의 채널로 확장해야 한다. 

```python
def create_time_channel(t, height, width, T):

    # 시간 정규화
    t_normalized = t.float() / (T - 1)

    # (B,) → (B,1,1,1)
    t_normalized = t_normalized.view(-1, 1, 1, 1)

    # (B,1,H,W)
    time_channel = t_normalized.expand(
        -1,
        1,
        height,
        width
    )

    return time_channel


# 테스트용 시간
t_test = torch.tensor(
    [0, 100, 500, 999],
    device=device
)

# Time Channel 생성
time_channel = create_time_channel(
    t_test,
    height=64,
    width=64,
    T=T
)

print("Time shape:", t_test.shape)
print("Time channel shape:", time_channel.shape)

print("Time values:")
print(time_channel[:, 0, 0, 0])
```

위에선 테스트로 time channel을 만들었다. 나중에 네트워크에서 다시 설명하겠지만, 이미지 tensor와 시간 tensor를 간단히 concat 만 하면 embedding이 된다. 

```python
# 예시코드, 이미지와 시간 채널 결합
model_input = torch.cat(
    [image, time_channel],
    dim=1
)
```

## CNN Noise Prediction model
이제 본격적으로 노이즈를 예측하는 모델을 만들어보자. 앞서 정의한 `creat_time_channel()`을 이용하자. 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class SimpleNoisePredictor(nn.Module):

    def __init__(self, T=1000):

        super().__init__()

        self.T = T

        self.network = nn.Sequential(

            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t):

        B, C, H, W = x.shape

        # Time Channel 생성
        time_channel = create_time_channel(
            t,
            H,
            W,
            self.T
        )

        # 이미지와 시간 정보 결합
        model_input = torch.cat(
            [x, time_channel],
            dim=1
        )

        # 노이즈 예측
        predicted_noise = self.network(model_input)

        return predicted_noise
```

모델을 생성하고 테스트해보자. 

```python
model = SimpleNoisePredictor(T=T).to(device)

print(model)

# 테스트 이미지
dummy_x = torch.randn(
    4, 1, 64, 64,
    device=device
)

# 테스트 시간
dummy_t = torch.tensor(
    [0, 100, 500, 999],
    device=device
)

# 모델 실행
output = model(dummy_x, dummy_t)

print("Input shape:", dummy_x.shape)
print("Output shape:", output.shape)
print("Output device:", output.device)
```
<figure>
  <img src="/assets/images/2026-10-20-02-18-55.png" style="width:80% !important; height:auto;" alt="2026-10-20-02-18-55">
  <figcaption>2026-10-20-02-18-55</figcaption>
</figure>


## CNN 학습

이제 학습에 들어가기 전, 데이터를 확인해보자. 

```python
print("Floorplans shape:", floorplans.shape)
print("Dataset size:", len(dataset))
print("Batch count:", len(dataloader))

sample_batch = next(iter(dataloader))[0]

print("Batch shape:", sample_batch.shape)
print("Min:", sample_batch.min().item())
print("Max:", sample_batch.max().item())
```
<figure>
  <img src="/assets/images/2026-10-20-02-19-52.png" style="width:80% !important; height:auto;" alt="2026-10-20-02-19-52">
  <figcaption>2026-10-20-02-19-52</figcaption>
</figure>

이제 Optimizer를 설정하고, 

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

loss_history = []

print("Training setup completed.")
```

학습을 실행해보자. 

``` python
model.train()

for epoch in range(epochs):

    total_loss = 0.0

    for batch in dataloader:

        # 원본 이미지
        x0_batch = batch[0].to(device)

        B = x0_batch.shape[0]

        # 시간 무작위 선택
        t = torch.randint(
            low=0,
            high=T,
            size=(B,),
            device=device
        )

        # 실제 Gaussian Noise
        noise = torch.randn_like(x0_batch)

        # Alpha Bar
        alpha_bar_t = alpha_bars[t].view(
            B, 1, 1, 1
        )

        # Forward Diffusion
        xt = (
            torch.sqrt(alpha_bar_t) * x0_batch
            +
            torch.sqrt(1 - alpha_bar_t) * noise
        )

        # CNN Noise Prediction
        predicted_noise = model(xt, t)

        # MSE Loss
        loss = F.mse_loss(
            predicted_noise,
            noise
        )

        # Gradient 초기화
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Parameter Update
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    loss_history.append(avg_loss)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.6f}"
    )
```
<figure>
  <img src="/assets/images/2026-10-20-02-21-14.png" style="width:80% !important; height:auto;" alt="2026-10-20-02-21-14">
  <figcaption>2026-10-20-02-21-14</figcaption>
</figure>

심플하다보니, 시간도 얼마 안걸린다. Loss 값은 실행 환경과 무작위 초기화에 따라 달라진다.

간단히 시각화해서 확인해보자..

```python
plt.figure(figsize=(8, 5))

plt.plot(
    loss_history,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Noise Prediction Training Loss")

plt.grid(True)
plt.show()
```
<figure>
  <img src="/assets/images/2026-10-20-02-22-15.png" style="width:80% !important; height:auto;" alt="2026-10-20-02-22-15">
  <figcaption>2026-10-20-02-22-15</figcaption>
</figure>


## 학습 결과 확인
이제 노이즈도 예측했으니, 원본이미지도 추정해보자.

```python
model.eval()

with torch.no_grad():

    # 테스트용 원본 이미지
    x0_test = floorplans[0:1].to(device)

    # Diffusion 시간
    t_test = torch.tensor(
        [300],
        device=device
    )

    # 실제 노이즈 생성
    noise = torch.randn_like(x0_test)

    # Alpha Bar
    alpha_bar_t = alpha_bars[t_test].view(
        1, 1, 1, 1
    )

    # Forward Diffusion
    xt = (
        torch.sqrt(alpha_bar_t) * x0_test
        +
        torch.sqrt(1 - alpha_bar_t) * noise
    )

    # CNN 노이즈 예측
    predicted_noise = model(
        xt,
        t_test
    )

    # 원본 이미지 추정
    estimated_x0 = (
        xt
        -
        torch.sqrt(1 - alpha_bar_t) * predicted_noise
    ) / torch.sqrt(alpha_bar_t)
```

이제 위의 예측을 통해 결과를 시각화 해보면 된다. 거의 다 왔다. 

<figure>
  <img src="/assets/images/2026-10-20-02-23-41.png" style="width:80% !important; height:auto;" alt="2026-10-20-02-23-41">
  <figcaption>2026-10-20-02-23-41</figcaption>
</figure>

위 이미지들은 다음과 같은 의미를 가진다. 

| 결과 | 의미 |
|---|---|
| Original | 원본 평면도 |
| Noisy Image | 노이즈가 추가된 이미지 |
| Actual Noise | 실제 노이즈 |
| Predicted Noise | CNN이 예측한 노이즈 |
| Estimated Original | 예측 결과를 이용한 원본 추정 |

비슷하게 나오는가? 이제 이렇게 원본 이미지에서 노이즈로 바꿔봤고, 이제 그 노이즈에서 다시 비슷한 이미지로 예측하는 과정을 실험해봤다. 
이 과정이 Diffusion의 과정이다. 


# 실습 4
지금까지 해본건 Diffusion 알고리즘에서 Forward 와 Reverse 를 통해 노이즈가 어떻게 병합되고 예측되는지 학습했고, 이 과정에서 CNN을 통해 노이즈를 예측하여 이를 반영해보았었다. 이제 해볼건 DDPM의 Reverse Sampling 과정을 통해 이미지가 어떻게 생성되는지 알아보려 한다. 

$$
x_0 \xrightarrow{\text{Forward Diffusion}} x_t
$$

$$
\epsilon_\theta(x_t,t)
$$

이번엔 예측한 노이즈를 이용해 역방항으로 노이즈를 단계적으로 제거하는 과정을 한다. 

## 수식들


DDPM에서는 다음과 같은 역방향 확률분포를 모델링한다. 

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N} \left(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2I\right)
$$

이때 평균은 다음과 같다. 

$$
 \mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)
$$

실제 샘플링에서는 평균에도 노이즈 PDF를 추가한다. 

$$
\boxed{ x_{t-1} = \mu_\theta(x_t,t)+\sigma_tz }
$$

$$
z\sim\mathcal{N}(0,I)
$$

그리고 마지막 단계에서는 추가 노이즈를 사용하지 않는다. 


위의 평균을 정의한 수식을 보면 $\beta_t$ 가 있는걸 알 수 있다. 현재단계의 노이즈 분산이지만, 이번 실습에선 Posterior Variance를 사용한다. 자연스럽게 $\alpha_t = 1-\beta_t$가 된다. Forward Variance 와 Posterior Variance 에 대해서는 따로 한번 깊게 파보도록 하겠고, 이번엔 맛보기 실습이니 그러려니 하자. 

수식은 다음과 같다. 

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$

## 환경점검

앞서 실습 3까지 잘 따라왔다면, 이 부분에서 큰 문제가 되는건 없을 것이다. 

실행환경과 모델의 상태를 점검해보자. 

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

required_variables = [
    "device",
    "T",
    "betas",
    "alphas",
    "alpha_bars",
    "model"
]

for name in required_variables:
    if name in globals():
        print(f"{name}: OK")
    else:
        print(f"{name}: NOT FOUND")

model = model.to(device)
model.eval()

print("Device:", device)
print("Diffusion Steps:", T)
print("Model:", type(model).__name__)

print(
    "Model Device:",
    next(model.parameters()).device
)
```
<figure>
  <img src="/assets/images/2026-10-20-13-43-37.png" style="width:80% !important; height:auto;" alt="2026-10-20-13-43-37">
  <figcaption>2026-10-20-13-43-37</figcaption>
</figure>
<figure>
  <img src="/assets/images/2026-10-20-13-43-49.png" style="width:80% !important; height:auto;" alt="2026-10-20-13-43-49">
  <figcaption>2026-10-20-13-43-49</figcaption>
</figure>


## Reverse Sampling 함수 구현
이제 실제 역확산함수를 구현해보자 .

```python
# 이전 단계의 alpha_bar
alpha_bars_prev = torch.cat(
    [
        torch.ones(1, device=device),
        alpha_bars[:-1]
    ]
)

# Posterior Variance
posterior_variance = (
    betas
    *
    (1 - alpha_bars_prev)
    /
    (1 - alpha_bars)
)

print("Posterior variance shape:", posterior_variance.shape)

print("First:", posterior_variance[0].item())
print("Last:", posterior_variance[-1].item())
```

<figure>
  <img src="/assets/images/2026-10-20-13-53-13.png" style="width:80% !important; height:auto;" alt="2026-10-20-13-53-13">
  <figcaption>2026-10-20-13-53-13</figcaption>
</figure>

첫 번째 값이 0인 이유는 마지막 역확산 단계에서 추가적인 노이즈가 필요하지 않기 때문이다. 

Reverse Sampling 함수를 표현하면 다음과 같다. 

```python
@torch.no_grad()
def reverse_step(model, xt, t):

    # 현재 시간
    t_tensor = torch.full(
        (xt.shape[0],),
        t,
        device=device,
        dtype=torch.long
    )

    # CNN 노이즈 예측
    predicted_noise = model(
        xt,
        t_tensor
    )

    # DDPM 계수
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]

    # Reverse Mean
    mean = (
        1 / torch.sqrt(alpha_t)
    ) * (
        xt
        -
        (
            beta_t
            /
            torch.sqrt(1 - alpha_bar_t)
        ) * predicted_noise
    )

    # 마지막 단계
    if t == 0:
        return mean

    # Gaussian Noise
    noise = torch.randn_like(xt)

    # Posterior Variance
    variance = posterior_variance[t]

    # Reverse Sampling
    x_prev = (
        mean
        +
        torch.sqrt(variance) * noise
    )

    return x_prev
```

이제 이 Reverse Sampling을 반복하는 loop를 구현하면 된다. 

```python
@torch.no_grad()
def sample_ddpm(
    model,
    image_size=64,
    num_samples=4
):

    model.eval()

    # 순수 Gaussian Noise
    x = torch.randn(
        num_samples,
        1,
        image_size,
        image_size,
        device=device
    )

    # 중간 결과 저장
    saved_images = {}

    save_steps = [
        999, 800, 600, 400,
        200, 100, 50, 0
    ]

    # Reverse Sampling
    for t in reversed(range(T)):

        x = reverse_step(
            model,
            x,
            t
        )

        if t in save_steps:
            saved_images[t] = x.detach().cpu().clone()

        if t % 100 == 0:
            print(f"Sampling Step: {t}")

    return x, saved_images
```

## 이미지 생성

이제 학습한 CNN을 사용해서 이미지를 생성해보자. 

```python
generated_images, saved_images = sample_ddpm(
    model,
    image_size=64,
    num_samples=4
)

print(
    "Generated images shape:",
    generated_images.shape
)
```
<figure>
  <img src="/assets/images/2026-10-20-14-20-16.png" style="width:80% !important; height:auto;" alt="2026-10-20-14-20-16">
  <figcaption>2026-10-20-14-20-16</figcaption>
</figure>


## 시각화

생성된 이미지를 확인해보자. 

```python
images = generated_images.detach().cpu()

plt.figure(figsize=(12, 3))

for i in range(4):

    plt.subplot(1, 4, i + 1)

    plt.imshow(
        images[i, 0],
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    plt.title(f"Generated {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
```

<figure>
  <img src="/assets/images/2026-10-20-14-21-01.png" style="width:80% !important; height:auto;" alt="2026-10-20-14-21-01">
  <figcaption>2026-10-20-14-21-01</figcaption>
</figure>

이게 뭔지 잘 모를 정도긴 하지만, 어찌되었건 뭔가 그럴싸한 흉내를 내려 애를 쓴 흔적은 보인다. 

중간단계들도 한번 시각화해보자. 

```python
steps = sorted(
    saved_images.keys(),
    reverse=True
)

plt.figure(figsize=(16, 8))

for i, step in enumerate(steps):

    img = saved_images[step][0, 0]

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        img.numpy(),
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    plt.title(f"t = {step}")
    plt.axis("off")

plt.tight_layout()
plt.show()
```

<figure>
  <img src="/assets/images/2026-10-20-14-22-01.png" style="width:80% !important; height:auto;" alt="2026-10-20-14-22-01">
  <figcaption>2026-10-20-14-22-01</figcaption>
</figure>

이런 과정으로 Diffusion이 동작하는구나 감만 잡아가자. 원본과 비교해보면 이렇다. 

```python
real_images = floorplans[:4].cpu()

generated_images_cpu = generated_images.cpu()

plt.figure(figsize=(12, 6))

for i in range(4):

    # 실제 데이터
    plt.subplot(2, 4, i + 1)

    plt.imshow(
        real_images[i, 0],
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    plt.title("Training Data")
    plt.axis("off")

    # 생성 데이터
    plt.subplot(2, 4, i + 5)

    plt.imshow(
        generated_images_cpu[i, 0],
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    plt.title("Generated")
    plt.axis("off")

plt.tight_layout()
plt.show()
```
<figure>
  <img src="/assets/images/2026-10-20-14-22-28.png" style="width:80% !important; height:auto;" alt="2026-10-20-14-22-28">
  <figcaption>2026-10-20-14-22-28</figcaption>
</figure>


여기까지 Diffusion이 이렇게 동작하는가에 대해 알아볼 수 있었다. 
물론 큰 틀의 개념적인 부분이고, 수식적으로 더 파고 봐야 할 부분도 있지만, Diffusion의 동작 개념과 방향이란 것에 대해 이렇구나 하고 알고만 넘어가자. 


# 정리
오늘 한걸 정리해보자. 

1. Diffusion Model의 기본 원리를 이해한다.
2. Forward Diffusion을 직접 구현한다.
3. 노이즈 예측 신경망을 학습한다.
4. Reverse Diffusion을 구현한다.
5. 랜덤 노이즈에서 이미지를 생성하는 전체 DDPM 과정을 경험한다.

위의 목표를 가지고 실습을 통해서 이미지 생성하는 것까지 수행했다. 

수식들과 $\alpha$, $\beta$ 와 같은 확률값도 언급했었다.  
노이즈 $\epsilon$ 를 생성해서 더하는 방법을 알아봤었고, 이 노이즈를 예측하기 위해 신경망을 썼다는걸 알수 있었다.  
이런 용어들에 대한 개념을 가지고, 좀더 상세한 실습으로 넘어가보자. 

오늘은 여기서 끗. 