---
#order: 70
layout: single
title: "ROS Beginner"
date: 2026-06-17 14:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
is_post: True
topic : ros
related: false
excerpt: "ROS Beginner"
tags:
  - SLAM
  - Robotics  
  - ROS
---

학습을 위한 페이지입니다. 오류 사항이나 지적해야 할 부분 있으면 편하게 연락주세요. 
민형기 강사님의 강의를 따라가는 페이지입니다. 
https://www.youtube.com/watch?v=aFMDvkCr9vY&list=PL0xYz_4oqpvhj4JaPSTeGI2k5GQEE36oi&index=36



# ROS의 설치
 
Ubuntu의 버전에 따라 사용하는 ROS의 버전이 다르다. 

<figure>
  <img src="/assets/images/2026-06-17-13-51-20.png" style="width:60% !important; height:auto;" alt="2026-06-17-13-51-20">
  <figcaption>2026-06-17-13-51-20</figcaption>
</figure>

이 PC는 Ubuntu 22.04이므로 Humble을 설치해본다. 


```bash
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe
```

ROS Key와 Repository를 등록한다. 

```bash
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
-o /usr/share/keyrings/ros-archive-keyring.gpg
```

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

ROS의 설치를 시작한다. 용량은 2~3GB라 한다. 설치하기 전에 확인하자. 
```bash
sudo apt update
sudo apt install ros-humble-desktop -y
```

환경변수를 등록하자. 

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

이제 설치된 ROS를 확인하자. 
```bash
printenv | grep ROS
```
<figure>
  <img src="/assets/images/2026-06-17-13-57-39.png" style="width:60% !important; height:auto;" alt="2026-06-17-13-57-39">
  <figcaption>2026-06-17-13-57-39</figcaption>
</figure>


ROS2 개발을 편하게 하기 위한 도구들도 설치한다. 
 - Colcon은 ROS2의 표준 빌드시스템이다. ROS1에선 catkin_make다. 
 - rosdep은 의존성 자동 설치도구다. 


```bash
sudo apt install python3-colcon-common-extensions -y
sudo apt install python3-rosdep -y
```

rosdep을 초기화 한다. 
```bash
sudo rosdep init
rosdep update
```

이제 정상작동하는지 확인해보자

- Terminal 1
```bash
ros2 run demo_nodes_cpp talker
```

- Terminal 2
```bash
ros2 run demo_nodes_py listener
```
<figure>
  <img src="/assets/images/2026-06-17-14-57-25.png" style="width:80% !important; height:auto;" alt="2026-06-17-14-57-25">
  <figcaption>2026-06-17-14-57-25</figcaption>
</figure>


이제 워크스페이스, Workspace를 생성하고 마무리지어보자. 

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build

source install/setup.bash
```

<figure>
  <img src="/assets/images/2026-06-17-15-41-09.png" style="width:80% !important; height:auto;" alt="2026-06-17-15-41-09">
  <figcaption>2026-06-17-15-41-09</figcaption>
</figure>

여기서 한가지 정도 체크하는게 좋다.   
ROS를 사용하는 사람이 늘면서, 가끔 네트워크상에서 ROS시스템간 메세지 충돌이 나는 경우도 발생한다.   
Domain ID를 기본값으로 설정하다보니, 거기서 발생하는 충돌이 원인인 경우다. 따라서 Domain ID를 확인하고 바꾸는 방법을 알아보자.  

```bash
echo $ROS_DOMAIN_ID
```

아무것도 설정된게 없는 경우 출력되는게 없다. 
이제 Domain ID를 설정해보자. 

```bash
export ROS_DOMAIN_ID=10
```

<figure>
  <img src="/assets/images/2026-06-17-15-56-11.png" style="width:60% !important; height:auto;" alt="2026-06-17-15-56-11">
  <figcaption>2026-06-17-15-56-11</figcaption>
</figure>

실행하면 위와 같이 나온다.

가급적 좀 독특한 자신만의 숫자를 쓰는게 나중을 위해서 좋을 것이다. 나는 84번으로 바꿔본다. (83번은 다른 노트북이라)   

<figure>
  <img src="/assets/images/2026-06-17-15-57-45.png" style="width:60% !important; height:auto;" alt="2026-06-17-15-57-45">
  <figcaption>2026-06-17-15-57-45</figcaption>
</figure>




---

# ROS2의 동작 구조/명령어들



<figure>
  <img src="/assets/images/2026-06-17-16-07-28.png" style="width:80% !important; height:auto;" alt="2026-06-17-16-07-28">
  <figcaption>2026-06-17-16-07-28</figcaption>
</figure>

<figure>
  <img src="/assets/images/2026-06-17-16-07-45.png" style="width:80% !important; height:auto;" alt="2026-06-17-16-07-45">
  <figcaption>2026-06-17-16-07-45</figcaption>
</figure>


## Node / Topic

ROS에서 실행되는 프로그램 하나를 Node라고 부른다. 
그리고 Node끼리 데이터를 계속 주고받는 통신 방식 중 가장 대표적인 것이 Topic이다.

먼저 가장 기본적인 통신은 방금전의 Terminal 1, 2에서 하고 있을 것이다. 
ROS 는 Node들로 구성하고, 이 노드들간에 Topic과 Message를 주고 받는 방식으로 동작한다. 

방금전, 우리는 talker라는 노드와 listener라는 노드를 실행했고, talker가 "Hello world"라는 \chatter 토픽을 보내서 listener가 받은 것을 화면에 출력하고 있는 것이다. 

Node와 topic을 보면 다음과 같다. 

<figure>
  <img src="/assets/images/2026-06-17-15-36-02.png" style="width:60% !important; height:auto;" alt="2026-06-17-15-36-02">
  <figcaption>2026-06-17-15-36-02</figcaption>
</figure>


그리고 토픽으로 무슨 내용이 가고 있는지 보고 싶으면 다음과 같이 실행하면 된다. 

```bash
ros2 topic echo /chatter
```

<figure>
  <img src="/assets/images/2026-06-17-15-38-44.png" style="width:80% !important; height:auto;" alt="2026-06-17-15-38-44">
  <figcaption>2026-06-17-15-38-44</figcaption>
</figure>



## Service

노드가 요청을 하면 다른 노드가 응답을 하는걸 서비스라 한다. 
한번만 통신한다. 


## Action 
어떤 goal 을 주어줬을때, 동작을하며 feedback을 회신하다가 최종 결과를 출력하는것까지 action이라 이해하자. 

이제 turtlesim을 이용해서 각각의 요소들을 한번 확인해보자. 

<figure>
  <img src="/assets/images/2026-06-17-16-19-29.png" style="width:80% !important; height:auto;" alt="2026-06-17-16-19-29">
  <figcaption>2026-06-17-16-19-29</figcaption>
</figure>
