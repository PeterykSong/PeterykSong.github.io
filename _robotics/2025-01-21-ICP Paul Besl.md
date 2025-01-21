---
#order: 70
#layout: post
title: A Method for Registration of 3D shape
date: 2025-01-21 21:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900

excerpt: "ICP 알고리즘 공부자료입니다. "
tags:
  - ICP
  - AI
  - Robotics
  - SLAM
---

이 페이지는 A method for registration of 3D Shape 논문 리뷰를 위해 작성했습니다. 
{:.notice}

2007년즈음으로 기억한다. 선배가 Stereo Vision을 이용해서 이동속도와 회전속도를 판단하는 알고리즘을 내게 보여준 적이 있었다. 그때 제일 먼저 궁금했던건, 좀전의 이미지에서 보이는 특징점이 그 다음 이미지의 특징점과 어떻게 매칭이 되는지였다. 그때도 아마 ICP라는 용어를 그 선배가 내게 설명했을법 한데, 알고리즘의 큰 의미는 이해했지만, 그것이 어떻게 세부적으로 돌아가는지에 대해서 당시에는 내가 이해할 능력이 못되었나보다. 

SLAM, 즉 로봇이 자신의 센서정보를 바탕으로 하여 현 시점의 환경정보와 Map을 생성했을때, 이를 가지고 있는 지도데이터와 비교하는 과정에서 ICP는 매우 핵심적인 역할을 한다. 현재의 지도를 천천히 이동/회전을 시켜 원래의 지도와 맞는 이동거리/회전을 구해준다. 이를 통해 원래 지도에서 나의 위치가 어디있는지 추정할 수 있는 수단을 제공한다. 

문득 실제 지도를 볼때 쓰는 방법과 비슷하다는 생각을 했다. 실제 지형을 보고, 지도를 빙빙 돌려가면서 내 위치를 찾는 방법 아닐까. 

이제 처음 SLAM관련 기본 논문을 읽어보는 입장에서, 이 이후에 얼마나 많은 발전이 있었을지 가늠할 수 없지만, 적어도 이 논문이 작성된 1998년도부터, 2000년대 초반까지는 매우 중요한 이정표같은 것이리라 생각이 든다. 

<div style="text-align: center;">  
![Image Registration](/assets/images/robotics/image_regitration.jpg)  
[이미지 Registration(정합)](https://kr.mathworks.com/discovery/image-registration.html)
</div>


이제 논문을 자세히 읽어내려가보자. 

# 논문의 목차
  1. Introduction
  2. 문헌 리뷰 (요새 이런걸 메타 연구라고 하나보다.)
  3. Preliminaries. (수학적 선수학습내용)  
      3.1. 기본공식식  
        3.1.1. 특정점과 점들간의 거리  
        3.1.2. 특정점과 선들과의 거리  
        3.1.3. 특정점과 삼각형들과의 거리 (Triangle을 계산하지만, 후에는 Polygon이 될듯)  
      3.2. 특정점과 함수객체와의 거리를 최소화(최적화)하는 방법 (Newton minimaztion approch를 사용함)  
      3.3. 특정점과 암시적(후보의)객체와의 거리 계산. 즉 위의 뉴턴 최소화방법을 사용한다.   
  4. ICP 알고리즘  
      4.1. 알고리즘의 설명  
      4.2. 수렴정리 (Convergence Theorem)  
      4.3. ICP알고리즘 가속화 방법  
      4.4. 다른 최소화 접근법  
  5. Set of Initial Registration  
     여기서부터는 좀 더 읽어봐야겠다.   
  6. 실험 결과  
  7. 결론  
  8. 향후 방향  

