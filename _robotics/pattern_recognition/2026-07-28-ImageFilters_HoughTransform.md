---
#order: 70
layout: single

title: "Image Filters - Hough Trasnform"
date: 2026-07-20 12:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
related: false
topic : pattern_recognition
excerpt: "ImageFilter, HoughTrasnform"
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


# 3. Hough Transform
 가우시안 블러도 그랬고, CannyEdgy Detection도 그렇듯, 특정한 행렬을 이용하여 그 픽셀과 주변과의 관계를 변환했다. 그 느낌 이어서, 이번엔 Hough Transform(허프 변환이라고 발음한다. 왜..? 휴도 아니고..)을 배워보자. 

 허프 변환은 이미지의 픽셀 공간에서 검출하기 어려운 기하학적 형태를 -> 매개변수 공간으로 변환하여 찾는 것을 말한다. 즉 kernel 을 이용해 다른 공간으로 변환시키는게 1번째 단계이고, 변환 이후의 관계를 이용해 원래 공간에서 기하학적 형태를 찾아가는 과정이다. 

 이 허프 변환은 직선, 원, 타원, 그 외에 수식으로 표현가능한 곡선을 검출할때 사용한다. 

 https://ko.wikipedia.org/wiki/%ED%97%88%ED%94%84_%EB%B3%80%ED%99%98

 수식적인건 다른 블로그들에서 상세히 설명하고 있으니 굳이 설명하려 하진 않는다.   
 큰 컨셉에서 이해를 먼저 한다면, 어렵지 않을듯하다.   

 워낙에 고전적인 방법이다보니 OpenCV에서는 이미 충분히 잘 지원해주고 있다.  

 알고리즘의 순서는 다음과 같다. 

 1. Canny Edge Detection 등의 방법으로 엣지 픽셀을 검출한다.   
 2. 각각의 엣지 픽셀에 대해 가능한 θ 값을 대입한다.
 3. 각 θ에 대응하는 ρ를 계산한다.
 4. 계산된 (ρ,θ) 위치의 누적값을 증가시킨다.
 5. 누적값이 큰 지점을 실제 직선의 후보로 판단한다.

투표 결과를 저장하는 배열을 누적 배열(accumulator array, $A(\rho,\theta) $ )이라 한다. 

$$
(\rho^\ast,\theta^\ast)
=
\underset{\rho,\theta}{\operatorname{arg\,max}}
\;A(\rho,\theta)
$$


(작성중) 수식은 차차 채워 나가보자. 


```python 
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 이미지 불러오기
image_path = Path("./data/rgb_rawlight.png")
image = cv2.imread(str(image_path))

result = image.copy()
h, w = image.shape[:2]

# 그레이스케일 변환 후 에지와 직선 검출
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 130)

line_count = 0 if lines is None else len(lines)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x0, y0 = cos_theta * rho, sin_theta * rho

        # 직선의 시작점에 원을 그려보자. 
        center = (int(round(x0)), int(round(y0)))
        if 0 <= center[0] < w and 0 <= center[1] < h:
            cv2.circle(result, center, 3, (0, 0, 255), -1)

        # 검출한 무한 직선이 이미지를 충분히 가로지르도록 끝점을 계산한다.
        length = max(h, w)
        x1 = int(round(x0 + length * (-sin_theta)))
        y1 = int(round(y0 + length * cos_theta))
        x2 = int(round(x0 - length * (-sin_theta)))
        y2 = int(round(y0 - length * cos_theta))
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)

print(f"검출된 직선 수: {line_count}")

# 결과를 확인해본다. 
fig, axes = plt.subplots(3, 3, figsize=(18, 7))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")

axes[1].imshow(edges, cmap="gray")
axes[1].set_title("Canny Edges")

axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title("Hough Lines")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()


```

검출된 직선 수는 254개다.   
시각화 해보면 결과는 이렇다... 개판인데..? 

<figure>
  <img src="/assets/images/2026-07-28-16-44-08.png" style="width:80% !important; height:auto;" alt="2026-07-28-16-44-08">
  <figcaption>2026-07-28-16-44-08</figcaption>
</figure>

이 많은 허프 변환을 다 하면 연산량이 많다. 

그래서 확율론을 적용시켜 검출하는 방법이 있다. 

```python
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, 20, 2)

for line in lines:
    # 검출된 선 그리기
    x1, y1, x2, y2 = line[0]
    cv2.line(img2, (x1,y1), (x2, y2), (0,255,0), 1)
```

그래도 좀 정제되어 보이긴 한다. 


<figure>
  <img src="/assets/images/2026-07-28-16-57-24.png" style="width:80% !important; height:auto;" alt="2026-07-28-16-57-24">
  <figcaption>2026-07-28-16-57-24</figcaption>
</figure>