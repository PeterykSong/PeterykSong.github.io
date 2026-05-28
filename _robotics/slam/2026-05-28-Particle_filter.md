---
#order: 80
layout: single
title: "Particle Filter, Fast SLAM"
date: 2026-05-28 09:00:00 +0900
excerpt: "고전 SLAM의 쌍벽. 그리고 지금도 강력한."
topic: slam
related: false

tags:
  - Computer Vision
  - YOLO
  - Robotics
  
---
# 시작하면서. 
  Particle Filter 는 고전 SLAM에서 가장 많이 썼던 SLAM의 쌍벽 중 하나다. 보통 고전 SLAM이라 하면 Filter 기반 SLAM이라고도 하는데, 하나는 Kalman Filter 기반의 SLAM이라면, 다른 하나가 Particle Filter 가 되시겠다. 

  Kalman Filter 는 Covariance(공분산)을 이용한 선형식에 기반하여 관측값, 자기 위치를 업데이트한다고 요약한다면, Particle Filter는 측정된 관측값들과 Monte-Carlo 기법을 이용하여 자신의 위치를 업데이트하는 전략을 취한다. 이 특성으로 인해 비선형적인 시스템에서 강력한것이 이 Particle filter다.   

  그래서 이론만 가지고 연구할땐 Kalman filter나 Particle Filter 둘 중 하나만 사용하는 경우가 많지만, 실무의 영역에선 두 필터를 경우에 따라 적절하게 섞어서 사용하는 경우가 많았다. 예를 들어 Local Map의 업데이트는 KF(Kalman Filter) 기반이라면, Global Map에서 로봇의 위치를 찾는데는 PF(Particle Filter)를 사용한다던가 하는 방식이다. 

  또 하나는 LiDAR 와의 궁합이 좋아서, LiDAR를 사용하는 로봇청소기 중 초창기 버전은 PF를 많이 사용했던 것으로 알고 있다. 물론 LiDAR에서도 KF 를 적용하여 사용할 수도 있다. 하지만, Global Map에서 자기위치를 찾을 수 있는 방법을 논할때, BoW(Bag of Words) 의 사용이 가능한 Vision SLAM에 대비해 2D-LiDAR은 이렇다 할 방법이 없었는데 이때 PF 기반의 알고리즘들이 강력한 대안이 되어준다. 

  지금은 Neural Network 기반의 알고리즘들도 종종 등장한다. 

  하지만, 제어 분야에 오래된 격언이 있다. 
  가장 기초적인 것도 되지 않는게 고급기능 넣는다고 그 성능이 드라마틱하게 바뀌지 않는다.   
  PID 로 되지 않는 놈이 SMC 넣는다고 제어가 되는게 아니다. (하는 사람도 있지만, 그건 천재의 영역이고)  
  PF조차 안되는놈이 NN쓴다고 갑자기 되는 경우는 없다..(고 믿고싶다.)


# 논문 읽어보기
  이제 고전을 하나 읽어보자. PF 가 유명해지기 시작한 그 첫번째 논문. 1999년 ICRA에서의 일이다. 

  F. Dellaert, D. Fox, W. Burgard and S. Thrun, "Monte Carlo localization for mobile robots," Proceedings 1999 IEEE International Conference on Robotics and Automation (Cat. No.99CH36288C), Detroit, MI, USA, 1999, pp. 1322-1328 vol.2, doi: 10.1109/ROBOT.1999.772544.

  https://ieeexplore.ieee.org/document/772544
  또는   
  https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/dellaert_frank_1999_2/dellaert_frank_1999_2.pdf

  <figure>
  <img src="/assets/images/2026-05-28-19-03-27.png" style="width:800px !important;"  alt="2026-05-28-19-03-27">
  <figcaption>Particle filter의 원문이다.</figcaption>
</figure>

  저자 하나하나 모두가 유명하고 레전드들이다. (SLAM분야에 있어서.)
  <figure>
  <img src="/assets/images/2026-05-28-19-06-40.png" style="width:600px !important;"  alt="2026-05-28-19-06-40">
  <figcaption>Frank Dellaert</figcaption>
</figure>


  <figure>
  <img src="/assets/images/2026-05-28-19-08-22.png" style="width:400px !important;"  alt="2026-05-28-19-08-22">
  <figcaption>Dieter Fox</figcaption>
</figure>

<figure>
  <img src="/assets/images/2026-05-28-19-09-56.png" style="width:400px !important;"  alt="2026-05-28-19-09-56">
  <figcaption>Wolfram Burgard</figcaption>
</figure>

<figure>
  <img src="/assets/images/2026-05-28-19-12-02.png" style="width:400px !important;"  alt="2026-05-28-19-12-02">
  <figcaption>Sebastian Thrun</figcaption>
</figure>

Sebastian Thrun, Wolfram Burgard, Dieter Fox 이 셋은 SLAM 학도들의 영원한 바이블 격인 Probabilistic Robotics 의 저자이다. 2010년대까지만 하더라도 로봇 전공한다고 하면 이 책을 한번쯤은 거쳐가야 했고, 로봇공학 전공하고 졸업했습니다 하는 사람의 책상에 꼭 한권씩은 끼워져있던 책이었다.(심지어 기계설계담당자 책상에도 꽂혀있더라.)
<figure>
  <img src="/assets/images/2026-05-28-19-14-09.png" alt="2026-05-28-19-14-09">
  <figcaption>Probabilistic Robotics </figcaption>
</figure>

이제는 시절이 좋아져서, 번역본도 등장했다. ........ 정말 좋은 시절이다.   
나땐 저거 배우다 영어의 문턱에서 좌절했었는ㄷ........  

<figure>
  <img src="/assets/images/2026-05-28-19-16-29.png" style="width:400px !important;" alt="2026-05-28-19-16-29">
  <figcaption>2026-05-28-19-16-29</figcaption>
</figure>


# Code Review

# 오늘 요약