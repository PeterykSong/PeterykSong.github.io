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


## 퀴즈. 
다음 명령어들을 설명해보자. 

- ros2 node list
- /turtle1/cmd_vel
- ros2 topic echo /turtle1/pose
- ros2 service call /clear std_srvs/srv/Empty
- /turtle1/rotate_absolute

---

# Publisher / Subscriber
이제 Python 코드로 ROS2 Topic 통신을 직접 만들어보자. 

앞서서 workspace를 준비한걸 기억해보자. 

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build

source install/setup.bash
```
src 안에 소스코드를 넣고, colcon build를 하면 build, install, log 폴더가 생긴다. 

```
ros2_ws/
├── src/       ← 내가 만든 패키지
├── build/     ← 빌드 중간 결과
├── install/   ← 실행 가능한 결과
└── log/       ← 빌드 로그
```

## Package 만들기. 
패키지(package)는 ROS 프로그램을 묶어놓은 프로젝트 폴더이다.
ROS가 관리하는 최소 단위의 프로젝트다. 
이제 소스 폴더로 가서, 패키지를 생성해보자. 

```bash
cd ~/ros2_ws/src
ros2 pkg create chapter4 --build-type ament_python --dependencies rclpy std_msgs
```

- chapter4는 패키지 이름
- --build-type ament_python은 Python 기반 ROS2 패키지를 만들겠다는 뜻
- --dependencies rclpy std_msgs는 이 패키지가 사용할 라이브러리

라는 뜻이다. 

| 항목         | 의미                             |
| ---------- | ------------------------------ |
| `rclpy`    | Python에서 ROS2 노드를 만들기 위한 라이브러리 |
| `std_msgs` | 기본 메시지 타입 모음                   |
| `String`   | 문자열 메시지 타입                     |

이렇게 생성하면, 다음과 같은 화면이 지나갈 것이다. 

<figure>
  <img src="/assets/images/2026-06-17-20-48-55.png" style="width:80% !important; height:auto;" alt="2026-06-17-20-48-55">
  <figcaption>2026-06-17-20-48-55</figcaption>
</figure>

<figure>
  <img src="/assets/images/2026-06-17-20-49-20.png" style="width:80% !important; height:auto;" alt="2026-06-17-20-49-20">
  <figcaption>2026-06-17-20-49-20</figcaption>
</figure>

 - package.xml : 패키지에 대한 설명서
 - setup.py : 실행등록파일 publisher나 subscriber등과 같은 정보를 기록한다. 


## Pubhlisher의 작성. 
Publisher는 메세지를 보내는 Node를 말한다.   
카톡에 비유한다면 "안녕하세요"를 계속 말하는 사람이다.   

이제 이 퍼블리셔를 만들어보자. 

```bash 
cd ~/ros2_ws/src/chapter4/chapter4
touch publisher.py
```
이러면 이제 publisher.py라는 파일이 만들어진다. 
touch 라는 명령어는 그냥 그러려니 하자. 저러면 만들어진다. 
이제 이 파일안에 필요한 코드를 넣어본다. 

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PublisherNode(Node):

    def __init__(self):

        super().__init__('publisher_node')
        self.publisher_ = self.create_publisher(
            String,
            'chatter',
            10
        )
        self.timer = self.create_timer(
            1.0,
            self.publish_message
        )
        self.count = 0

    def publish_message(self):
        msg = String()
        msg.data = f'Hello ROS2 {self.count}'
        self.publisher_.publish(msg)
        self.get_logger().info(msg.data)
        self.count += 1


def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

```

코드는 단순하다. 1초마다 Hello ROS2 {self.count} 를 생성해서 '/chatter' topic으로 전송한다. 

상세하게 코드를 뜯어보면, 

```python
self.publisher_ = self.create_publisher(
    String,
    'chatter',
    10
)
```
Sting 타입의 데이터를, chatter라는 Topic에 보내겠다 라는 뜻이다. chatter는 키워드가 아니고 그냥 만든 토픽이름이다. 

```python
        self.timer = self.create_timer(
            1.0,
            self.publish_message
        )
        self.count = 0
```
ROS2에게 "1초마다 publish_message()를 호출해줘" 라고 등록하는 것이다. 

그럼 실질적으로 메세지는 이렇게 만들어진다. 

```python
def publish_message(self):
    msg = String()  #from std_msgs.msg import String에서가져온 ROS 표준 문자열 메세지타입이다. 이런식으로 객체로 만들어야 한다. 
    msg.data = f'Hello ROS2 {self.count}' #메세지에 내용을 채우고. 
    self.publisher_.publish(msg) #토픽으로 전송한다. 
    self.get_logger().info(msg.data) #이건 화면에 출력하는 역할을 한다. 
    self.count += 1
```


## Subsriber의 작성

이제 반대편 노드인 Subscriber를 만들자. 

```python 
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SubscriberNode(Node):

    def __init__(self):

        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.callback,
            10
        )
    def callback(self, msg):
        self.get_logger().info(
            f'Received : {msg.data}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SubscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

이제 구분지어서 다시 보자. 
subscription 인스턴스(객체)는 chatter 토픽을 구독하고, 메시지가 들어오면 callback() 함수를 실행하라는 뜻이다. 

```python
self.subscription = self.create_subscription(
            String,   #들어오는 토픽의 데이터타입이다. 
            'chatter',  #토픽 이름
            self.callback,  #토픽의 메세지를 수신하면 호출할 함수. 그래서 callback이다
            10  #이건 Queue의 크기다. 아.... 큐라 이야기 하는 순간, 이게 사람 여럿 잡았겠구나 싶다. 
        )

```

subsription 인스턴스가 선언되었으면, 
이제 callback함수를 보자. 별건 없다. 

```python
def callback(self, msg):
    self.get_logger().info(
        f'Received : {msg.data}'
    )
```
앞에서 봐서 알겠지만, 그냥 로거만 켜진다. 즉 화면에 뿌리기만 하고 있다. 


## 실행 등록
ROS는 자동으로 Python 파일을 실행하지 않는다. 
그래서 setup.py에 등록해야 한다. 

원래의 setup.py파일을 보자. 

```python
from setuptools import find_packages, setup

package_name = 'chapter4'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lairpeteryksong',
    maintainer_email='peteryksong@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={   #여기다 여기가 
        'console_scripts': [
        ],
    },
)
```

아래에 보면 entry point가 있다. 여기에 실행되는 노드들을 넣는다. 

```python
entry_points={
    'console_scripts': [
        'publisher = chapter4.publisher:main',
        'subscriber = chapter4.subscriber:main',
    ],
},
```

이 뜻은 ros2 run chapter4 publisher 로 실행이 되면, main()이 실행된다는 의미다. 



## 빌드
여기까지 다 만들었으면 이제 빌드를 해보자. 

```python
cd ~/ros2_ws
colcon build
source install/setup.bash
```
<figure>
  <img src="/assets/images/2026-06-17-23-01-49.png" style="width:80% !important; height:auto;" alt="2026-06-17-23-01-49">
  <figcaption>2026-06-17-23-01-49</figcaption>
</figure>

중간에 한번 에러가 떴는데, python setuptools 버전충돌이 있었다. 중간중간에 이런것들이 있네.   

만약 빌드가 자꾸 에러가 뜬다거나 할때, 빌드 폴더를 비워야 할 일이 있을 수도있다. 

```bash
cd ~/ros2_ws
rm -rf build install log
```

이제 실행을 해보자. 

터미널 1에선 퍼블리셔를
```bash
ros2 run chapter4 publisher
```

터미널 2에선 받는쪽을
```bash
ros2 run chapter4 subscriber
```

<figure>
  <img src="/assets/images/2026-06-17-23-06-21.png" style="width:80% !important; height:auto;" alt="2026-06-17-23-06-21">
  <figcaption>2026-06-17-23-06-21</figcaption>
</figure>



# Service sever/client

방금전깢, 메세지를 꾸준히 보내는, 즉 Topic을 보내고(퍼블리시) 받는(Subscrib) 구조를 했다. 
이번엔, 한번만 핑~~ 보내고 받는 서비스 서버/클라이언트를 해보려 한다. 이것까지 하면 일단 ROS는 이렇게 되어있구나 알 수 있다. 

## 실습구조. 
Turtlesim이 동작중이라고 가정한다.   
하나 띄워두자.  

<figure>
  <img src="/assets/images/2026-06-18-00-45-03.png" style="width:80% !important; height:auto;" alt="2026-06-18-00-45-03">
  <figcaption>2026-06-18-00-45-03</figcaption>
</figure>

이미 /spawn이란 서비스 서버를 제공하지만. 
커스터마이징 된 /spawn_client 를 하나 만들어보자. 
파일 명은 spawn_client.py로 하자. 
저장은 퍼블리셔가 있는 chapter4폴더에 넣자. 

```python
import rclpy
from rclpy.node import Node
from turtlesim.srv import Spawn


class SpawnClient(Node):
    def __init__(self):
        super().__init__('spawn_client')
        self.client = self.create_client(
            Spawn,
            '/spawn'
        )

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                'waiting for service...'
            )

    def send_request(self):
        request = Spawn.Request()
        request.x = 5.0
        request.y = 5.0
        request.theta = 0.0
        request.name = 'new_turtle'
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future
        )

        return future.result()

def main(args=None):
    rclpy.init(args=args)
    node = SpawnClient()
    response = node.send_request()
    node.get_logger().info(
        f'Spawned: {response.name}'
    )
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

```

이 코드가 하는 일은 다음과 같다. 

1. /spawn 서비스 찾기
2. 거북이 생성 요청
3. 서버 응답 기다리기
4. 생성된 이름 출력
5. 종료

각각의 인스턴스는 다음과 같다. 

### Client 생성

```python
self.client = self.create_client(
    Spawn,
    '/spawn'
)
```
앞서서 퍼블리셔에서는 create_publisher()였다면 이번엔 client를 만든다. 
이 클라이언트는 "나는 Spawn 타입의 /spawn서비스와 통신할 것이다" 라고 선언하는 것이다. 

Spawn 은 `from turtlesim.srv import Spawn` 여기서 선언되었다. 

### Service 존재 확인

```python
while not self.client.wait_for_service(
    timeout_sec=1.0
):
```
혹시나 클라이언트가 먼저 실행될 경우를 대비하기 위해서 있는 코드.

### Request 생성
```python 
request = Spawn.Request()
```

앞서 퍼블리셔에서는 `msg = String()` 였었다. 동일한 컨셉으로 이해하면 된다. 
Service 요청 객체를 선언한다고 보면된다. 그게 request다. 

### 요청 데이터 입력

```python
request.x = 5.0
request.y = 5.0
request.theta = 0.0
request.name = 'new_turtle'
```
이제 보낼 내용을 정리해보자. 
만약 뭘 써야 할지 모른다면, interface 명령을 한번 써보는것도 좋다. 

<figure>
  <img src="/assets/images/2026-06-18-01-00-02.png" style="width:80% !important; height:auto;" alt="2026-06-18-01-00-02">
  <figcaption>2026-06-18-01-00-02</figcaption>
</figure>

이미지에서도 보이지만, float형태 변수 세개인데, x,y,theta가 넘어가야 하고. 이름은 옵션이라고 한다. 

위의 코드에선 (5,5)라는 위치에 거북이를 생성해달란 소리다. 

### 요청 전송
이제 request를 보내보자. 

```python
future = self.client.call_async(
    request
)
```
앞서 퍼블리시에서는 `publish(msg)`를 사용했는데, 이번엔 `call_async(request)`를 사용한다. 

### 응답대기
이제 요청을 보내고, 그게 잘 되었는지 기다리기만 하면 된다. 

```python
rclpy.spin_until_future_complete(
    self,
    future
)
```
결과는 response에 저장된다. 
나머지는 로그를 출력한다던가 그런 부분이니 넘어간다. 

### 실행등록. 

마찬가지로 노드를 만들었으니, 패키지에 넣어 실행 가능하도록 등록해야 한다. 

setup.py를 찾아서 엔트리 포인트를 수정하자. 

```python
entry_points={
    'console_scripts': [
        'publisher = chapter4.publisher:main',
        'subscriber = chapter4.subscriber:main',
        'spawn_client = chapter4.spawn_client:main',
    ],
},
```

### 실행

source install/setup.bash 때문에 왠만하면 창을 닫고 다시 여는것도 좋다. 
만들었던 client가 어떻게 동작하는지 보자. 


터미널1
```bash
ros2 run turtlesim turtlesim_node
```

터미널2
```bash
ros2 run chapter4 spawn_client
```
<figure>
  <img src="/assets/images/2026-06-18-01-14-24.png" style="width:80% !important; height:auto;" alt="2026-06-18-01-14-24">
  <figcaption>2026-06-18-01-14-24</figcaption>
</figure>

색깔은 그때그때마다 바뀔 수 있다만. 일단 위의 이미지에서 녹색 거북이 옆에, 노란 거북이(5,5)위치에 하나 더 생긴걸 볼 수 있다. 클라이언트를 실행한 터미널창에서도 생성된걸 확인했다는 로그가 떴다. 

이런식으로 서비스/서버를 만들수 있다. 


이 구조 잘 쓰면, 방안의 전등을 원격으로 켜고 끌 수 있으려나...? 

여기까지 입문편 끝. 


### 번외
신기한것.
아까 스폰한것 한번 더하면 어떨까 싶어서, 더 눌러봤다.   

결과는 name이 같아서, 에러가 뜬다. 

즉, client 만들때도 이런 류의 주의가 필요하다. 

name에 뭔가 iter를 넣어야 하겠네. 아니면 랜덤값이라도. 



### 번외2. 
ROS 노드를 설계할때, 이런 설계 구조를 만들어두는 것도 좋을 것 같다.   
요새같은 AI시대에, AI한테 코드 작성을 맡기려면 이정도 명세서는 필요하겠지..   

```
MyNode

Publisher
 └─ /cmd_vel

Subscriber
 └─ /scan

Timer
 └─ 10Hz

Service
 └─ /reset
 ```

이번 예제를 가지고 표로 만들어보면 이렇게 되겠다.    

| 항목           | 내용                |
| ------------ | ----------------- |
| Node 이름      | `publisher_node`  |
| Publish      | `/chatter`        |
| Message type | `std_msgs/String` |
| Timer        | `1.0 sec`         |
| 상태 변수        | `count`           |

ROS2의 노드 구조는 아래와 같은 형태를 띈다. 

```python
import rclpy
from rclpy.node import Node


class MyNode(Node):  #Node가 반드시 들어간다. import Node의 속성을 상속하기 때문. 

    def __init__(self):
        super().__init__('my_node')  #ROS 노드의 이름을 등록한다. 

        # 1. 상태 변수
        self.count = 0

        # 2. Publisher
        self.publisher_ = self.create_publisher(
            MsgType,
            'topic_name',
            10
        )

        # 3. Subscriber
        self.subscription = self.create_subscription(
            MsgType,
            'topic_name',
            self.callback,
            10
        )

        # 4. Service Client / Server
        # self.client = self.create_client(...)
        # self.service = self.create_service(...)

        # 5. Timer
        self.timer = self.create_timer(
            1.0,
            self.timer_callback
        )

    def callback(self, msg):
        # Subscriber가 메시지를 받았을 때 실행
        pass

    def timer_callback(self):
        # Timer 주기마다 실행
        pass


def main(args=None):
    #ROS 노드의 실행과 종료를 설정하는 표준코드다. 
    rclpy.init(args=args)   #ROS2 Python 라이브러리를 시작한다. 

    node = MyNode()         #노드 객체를 생성하는 작업이다. MyNode.__init()__이 실행된다. 

    rclpy.spin(node)  #노드의 실행상태를 유지한다. 그래서 spin()함수를 사용한다. 없으면 종료된다. 

    node.destroy_node() #노드의 리소스를 반환/해제한다. 
    rclpy.shutdown() #ROS2 python시스템을 종료한다. 


if __name__ == '__main__':
    main()
```

위의 주석이 있는 요소들은 필수도 있고 비필수도 있으나, 어지간하면 넣어두자. pass가 될 지언정. 