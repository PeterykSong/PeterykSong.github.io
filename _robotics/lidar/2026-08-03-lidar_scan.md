---
#order: 70
layout: single

title: "LiDARs - 2D Simulaion LiDAR"
date: 2026-07-20 12:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
related: false
topic : lidar
excerpt: "LiDARs - 2D Simulaion LiDAR"
tags:
  - Robotics
  - SLAM
  - LiDAR
  
---

개인적으로 공부해가는 과정입니다. 

{: .notice}

# 들어가면서.   

LiDAR 관련된 내용들을 업데이트해보려 한다. 쉽고 유치한것부터, LiDAR 베이스 SLAM알고리즘 하나하나 한번 구현해보는 칸이 되려 한다. 잘 될진 모르겠지만. 


# 가상 LiDAR

Map의 어떤 지점에 2D LiDAR 가 있다고 할때, 센서의 결과값을 예측해보려 한다.   
간단한 알고리즘 연습이라 생각해도 될 것 같다. 

Unity 같은 게임엔진에선 Physics.Raycast 라는 명령어로 지원되지만, 만약 내가 원하는 데이터셋을 만들고자 할때, Unity를 사용한다는 보장은 없으니까. 

<figure>
  <img src="/assets/images/2026-08-03-13-09-09.png" style="width:80% !important; height:auto;" alt="2026-08-03-13-09-09">
  <figcaption>2026-08-03-13-09-09</figcaption>
</figure>

위와 같은 맵이 있다고 가정하자. 저 맵의 중간지점에서 360도 LiDAR를 돌렸을때 어떤 결과가 나올 것인가. 

- LiDAR는 1도 간격으로 적외선 레이저를 발신하여 돌아오는 값을 바탕으로 거리값을 측정할 것이다. 

그러면, 각 각도별로 해당하는 pixel을 찾아서 계속 스캔해나가면서 할 수도 있겠다. 


```python 
#아래 내용은 그냥 머릿속 손코딩 수준이다. 된다고 생각하지 말것. 
#헤더
import cv2
import math

#이미지 불러들이고 전처리
img = cv2.imread("map.png")
width, height = img.shape[:2]

#각도에 해당하는 픽셀 좌표값들 계산
bearing = 0 #방위각이니 바뀔수 있다고 보자. 
limit = 0 #탐지 거리

for i in range(limit):
  x = i * cos(bearing) + width/2
  y = i * sin(bearing) + height/2
  occupancy = img[x,y]

#픽셀이 차 있는지 확인
if occupancy == 1:
  lidar_data [bearing] = occupancy

```
요롷게 하면, 각도방향으로 주욱 직진하면서 occupancy를 읽어오니까. 어떻게든 된다. 
자.. 이거 실행하면 수많은 에러가 뜰 것이다. 된다고 생각하지 말라 했다. 

우선 sin/cos을 쓰니가, import math는 from math imoport sin, cos 으로 바뀌어야겠다.   
또한 이미지가 RGB (정확히는 BRG)인 가능성을 배제하기 위해 img = cv2.imread("map.png", cv2.IMREAD_GRAYSCALE)스타일로 처리하면 좋다.  

그 다음, OpenCV는 이미지를 읽어들일때 H X W 로 읽어드린다.  
그러니 height, width = img.shape[:2] 로 바꿔야 한다.   

방위각은 라디안으로 받아야 하니까, bearing = (라디안) 으로 해야 한다.   
따라서 bearing_deg 로 따로 정의를 하거나, 입력할때 랃디안으로 잘 변환해서 넣어야 한다.   
만약 import math를 했다면, bearing = math.radians(bearing_deg) 로 입력이 가능하다. 그게 아니라면. radian = deg/180*pi() 을 180도 이내와 이상으로 나누어 계산해야 한다. radians계산 편하네...  

읽어들일 x, y 좌표는 구했지만, float 형이다. img[x,y] 의 x, y는 인덱스값이므로 정수형에 맞춰줘야 한다. px = int(round(x)), py = int(round(y)) 로 바꿔서 img[px,py] 로 읽어보자.   
그런데, OpenCV에서는 H, W 순이므로, img[py,px]가 되어야 할 것이다.   
만약 py, px값이 이미지 밖에 있다면 break를 거는게 연산량을 줄이기 좋다. 
if px < 0 or px >= width or py < 0 or py >= height: break  

해당 픽셀의 값은 0과 1 범위에서 결정되지 않는다. 0~255사이값이므로, occupancy 를 비교할땐 if occupancy > 128 :  이런 형태로 구성하는게 맞다. (지금은 검은색이 공간이고 흰색이 벽이므로.)

이후 for 문으로 bearing 값을 올려가면서 순회한다면, 얼추 원하는 모양새가 만들어진다. 

입력과 출력을 명확히 해서 코드를 다시 정리하면. 

```python
#아래 내용은 그냥 머릿속 손코딩 수준이다. 된다고 생각하지 말것. 
#헤더
import cv2
from math import sin, cos, radians

# Input / 이미지 로드
img = cv2.imread("map.png", cv2.IMREAD_GRAYSCALE)

height, width = img.shape[:2]

# LiDAR 위치
sensor_x = width // 2
sensor_y = height // 2

# 방위각
bearing_deg = 45
bearing_rad = radians(bearing_deg)

# 최대 탐지 거리: 픽셀 단위
limit = 300

#Output
lidar_data = {}

# 장애물을 찾지 못하면 최대 거리로 설정
detected_distance = limit

for i in range(1, limit + 1):
    x = sensor_x + i * cos(bearing_rad)
    y = sensor_y + i * sin(bearing_rad)

    px = int(round(x))
    py = int(round(y))

    # 이미지 바깥으로 나간 경우
    if not (0 <= px < width and 0 <= py < height):
        detected_distance = i
        break

    occupancy = img[py, px]

    # 흰색을 장애물로 가정
    if occupancy > 128:
        detected_distance = i
        break

lidar_data[bearing_deg] = detected_distance

print(lidar_data)
```
이걸 이제 for문으로 전체 각도를 순환해보자. 
코드가 좀 길어지겠지만, 재사용을 위해 두개의 함수로 구분해보자. 한 각도에서 찾는 것을 cast_ray 함수라 하고, 이걸로 360도 돌리는걸 lidar_scan 이라는 함수로 정의해보자. 

```python

import cv2
from math import sin, cos, radians

def cast_ray( img, sensor_x, sensor_y, bearing_deg, max_range_px, obstacle_threshold=128, ):
   height, width = img.shape[:2]
    bearing_rad = radians(bearing_deg)

    for distance_px in range(1, max_range_px + 1):
        px = sensor_x + int(round(distance_px * cos(bearing_rad)))
        py = sensor_y + int(round(distance_px * sin(bearing_rad)))  #img의 y축 시작점이 위에서 시작한다. 

        # 이미지 범위를 벗어난 경우
        if not (0 <= px < width and 0 <= py < height):
            return distance_px

        occupancy = img[py, px]

        # 흰색 영역을 장애물로 가정
        if occupancy > obstacle_threshold:
            return distance_px

    # 최대 탐지 거리 내에 장애물이 없음
    return None


def simulate_lidar( img, sensor_x, sensor_y, max_range_px, obstacle_threshold=128,):
    lidar_data = {}

    for bearing_deg in range(360):

        distance = cast_ray(
            img=img,
            sensor_x=sensor_x,
            sensor_y=sensor_y,
            bearing_deg=bearing_deg,
            max_range_px=max_range_px,
            obstacle_threshold=obstacle_threshold,
        )

        lidar_data[bearing_deg] = distance

    return lidar_data  

# 이미지 로드
img = cv2.imread("map.png", cv2.IMREAD_GRAYSCALE)

height, width = img.shape[:2]

# LiDAR 센서를 이미지 중심에 배치
sensor_x = width // 2
sensor_y = height // 2

lidar_data = simulate_lidar(
    img=img,
    sensor_x=sensor_x,
    sensor_y=sensor_y,
    max_range_px=300,
    obstacle_threshold=128,
)

```

이 결과를 확인하기 위해 plot 을 작성해보자. 

```python
import numpy as np
import matplotlib.pyplot as plt

angles = np.radians(list(lidar_data.keys()))
distances = list(lidar_data.values())

ax = plt.subplot(projection="polar")
ax.plot(angles, distances)
plt.show()
```

<figure>
  <img src="/assets/images/2026-08-03-18-32-35.png" style="width:80% !important; height:auto;" alt="2026-08-03-18-32-35">
  <figcaption>2026-08-03-18-32-35</figcaption>
</figure>

음... 뭔가 각도가 바뀌었다... y축의 원점과 x축의 원점이 맞지 않는 문제가 있다. 

'py = sensor_y + int(round(distance_px * sin(bearing_rad)))' 의 +를 -로 바꿔서 실행해보면 바로된 결과를 얻을 수 있다.

<figure>
  <img src="/assets/images/2026-08-03-18-44-46.png" style="width:80% !important; height:auto;" alt="2026-08-03-18-44-46">
  <figcaption>2026-08-03-18-44-46</figcaption>
</figure>


이제, map에서 lidar값을 추정해볼 수 있겠다. 