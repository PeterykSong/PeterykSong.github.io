---
#order: 70
layout: single
title: "ROS Applying"
date: 2026-06-23 14:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
is_post: True
topic : ros
related: false
excerpt: "ROS 응용"
tags:
  - SLAM
  - Robotics  
  - ROS
---

학습을 위한 페이지입니다. 오류 사항이나 지적해야 할 부분 있으면 편하게 연락주세요. 
민형기 강사님의 강의를 따라가는 페이지입니다. 
https://www.youtube.com/playlist?list=PL0xYz_4oqpvg8qoYRQEIfQziTCMoTszrb



# 지난시간 복습. 

  지난시간에 Topic이 뭔지, Node가 뭔지, 그리고 Sevice와 Action이 뭔지 한번 알아봤었다. 

  다시한번 돌려보자. 

<figure>
  <img src="/assets/images/2026-06-23-13-35-29.png" style="width:80% !important; height:auto;" alt="2026-06-23-13-35-29">
  <figcaption>2026-06-23-13-35-29</figcaption>
</figure>


`create_publisher`, `create_subscriber`, `create_timer`를 사용해봤고,   
퍼블리셔인 경우, `self.publisher_.publish(msg)`로 메세지를 보내봤고,   
서브스크라이버에서, `callback`함수를 통해 msg를 받았을때의 핸들링을 만들어봤다. 

`main`에서는 `rclpy`를 초기화 하고, node를 생성/소멸하는 방법을 해봤다. 

client를 만들어봤었고, client class를 생성하고, main에서 서비스로 어떻게 request 하는지를 해봤었다. request 할 데이터를 미리 채우고 `request = Spawn.Request()`, `self.client.call_async(request)`를 통해 비동기 리퀘스트를 보냈다. 

잘 떠올려보자. 

# 오늘 해볼 것. 

 - ROS2 패키지 제작  
 - Custom Message 제작
 - Topic 응용
 - Service
 - Action
 - Parameter
 - Launch
 - rosbag
 - rqt

 많기도 하다. 하긴, 좀전의 강의는 4시간 짜리였는데, 이번건 6시간에 가깝다. 해야 할게 많다.   



 # Chapter 5. ROS2 패키지 제작.

ROS2에서 모든 코드는 패키지 단위로 관리된다. 즉, node에 해당하는 코드들을 쓰더라도, 패키지 등록을 해야 실행할 수 있는 프로그램이 된다. 이것은 setup.py설정을 하면서 한번 경험했었고, 빌드하는 과정을 통해 먼저 경험해봤었다. 기억을 떠올려라. 

<figure>
  <img src="/assets/images/2026-06-23-15-13-46.png" style="width:80% !important; height:auto;" alt="2026-06-23-15-13-46">
  <figcaption>2026-06-23-15-13-46</figcaption>
</figure>

여기에 보이는 폴더 하나하나가 패키지라는 의미다.   

패키지는 크게 두가지 역할을 한다. 
  - 코드의 보관 : Node, Topic, 서비스 등에 관련된 코드들
  - 메타정보의 관리 : 이 패키지 이름이 뭐고, 어떤 의존성이 있는지 등등. 

이제 이 패키지를 만들어보자. 

## 패키지 생성

이제 Chapter5니까. 패키지 이름을 chapter5로 해보자.

```bash
cd ~/ros2_ws/src
ros2 pkg create chapter5 --build-type ament_python
```

만들기 참 쉽죠?  
`ros2`를 실행해서, `pkg`관련 명령어 중, `create`를 실행하는데, 이름은 `chapter5`이고, 빌드 타입은 파이썬타입이므로, `ament_pyhon`으로 설정한다 는 뜻이다. 

<figure>
  <img src="/assets/images/2026-06-23-15-17-57.png" style="width:80% !important; height:auto;" alt="2026-06-23-15-17-57">
  <figcaption>2026-06-23-15-17-57</figcaption>
</figure>

그러면, 라이선스 정보들이나 버전정보들과 같은 메타정보들과 함께 패키지가 하나 만들어진다. 

만들어진 결과는 이렇다. 

<figure>
  <img src="/assets/images/2026-06-23-15-41-54.png" style="width:80% !important; height:auto;" alt="2026-06-23-15-41-54">
  <figcaption>2026-06-23-15-41-54</figcaption>
</figure>

자 이제 만들어진 파일을 자세히 훑어보자. 

## package.xml

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>chapter5</name>
  <version>0.0.0</version>
  <description>TODO: Package description</description>
  <maintainer email="peteryksong@gmail.com">lairpeteryksong</maintainer>
  <license>TODO: License declaration</license>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```
 - `<package format="3">` : 패키지 문서의 시작을 알린다. format은 문법 버전을 말하며, 3로 하면 문제 없다. ROS1 때 1이었다고 한다.    
 - `<name>chapter5</name>` : 패키지 이름이다. 나중에 `ros2 run chapter5 my_node`에서 그 이름이 나온다. 
 - <test_depend> 테스트시 필요한 의존성 패키지들을 확인하는 부분이다. colcon test 에서 필요한 부분인데... 잘 쓰려나..? 

 어쨌건 지금은 아무런 노드가 만들어진게 없기 때문에 `<depend>` 가 없다. `<depend>`는 공통의 의존성 패키지이므로, 커스텀 패키지를 만들면 언젠가는 쓰게 될 것 이다. 알고 가자. 

 | 태그                      | 의미                      |
| ----------------------- | ----------------------- |
| `<depend>`              | 빌드 및 실행에 필요한 의존성        |
| `<test_depend>`         | 테스트에만 필요한 의존성           |
| `<build_depend>`        | 빌드 시에만 필요               |
| `<exec_depend>`         | 실행 시에만 필요               |
| `<build_export_depend>` | 다른 패키지가 이 패키지를 사용할 때 필요 |


## setup.py
Python 패키지 설치 스크립트다. 
가장 중요한 부분은 지난 시간에도 말했지만 이 부분이다. 

```python
entry_points={
    'console_scripts': [
    ],
},
```

저기 위에 [] 괄호 안에 `'hello = chapter5.hello_node:main',` 를 넣으면, `ros2 run chapter5 hello`를 실행시킬 수 있게 해준다. 


만약 `hello_node.py`라는 파일이 chapter5안에 있다고해보자. 
그 파일을 다음과 같은 스크립트로 구성한다고 했을때(노드의 최소 구성요소를 알수 있다.)  

```python
from rclpy.node import Node

class HelloNode(Node):
    pass

def main():
    pass
```

이렇게 하면, `ros2 run chatper5 hello`를 실행하면, 방금의 `hello_node.py`파일 안에 있는 main이 실행되게 하는 것이다. 

## 그외에

resource 폴더에 보면 chapter5 라는 빈 파일이 하나 보인다. 
이게 있어야 ROS2가 여기 chapter5라는 패키지가 존재함을 알아차릴 수 있다. 그대로 놔두자. 

`__init__.py` 라는 파일도 눈에 보인다. 
이 폴더는 패키지라는 것을 의미한다. 
대부분은 비워두니 가만 두자. 

## concon build
이제 ROS2가 쓸수 있도록 컴파일을 해야 한다. 
컴파일의 과정은 아래와 같다. 

package.xml 확인 -> setup.py 확인 -> 의존성 확인 -> Python 패키지 등록 -> install 폴더 생성

빌드를 해보면, build, install, log 폴더가 생성되는걸 볼 수 있다. 

<figure>
  <img src="/assets/images/2026-06-23-21-00-45.png" style="width:80% !important; height:auto;" alt="2026-06-23-21-00-45">
  <figcaption>2026-06-23-21-00-45</figcaption>
</figure>

## 빌드가 끝났다면, setup.bash를

항상 빌드 후에, 
```bash
source install/setup.bash
```
해야 한다. 그래야 현재 터미널에서 ROS2가 빌드된 패키지들을 인식할 수 있다. 
그래서 어떤분들은 colcon build와 source setup.bash를 아예 배치로 만들어서 한번에 실행시키도록 셋팅하는 분들이 있을 정도다. (어떤 강의자료는 그렇게 하더라만.)

여기까지가, 지난시간에 은연중에 지나갔던 패키지 만들기 기본이었다. 

이제 커스텀 패키지/메세지를 제작해보자. 


# Chapter 6. Custom Message 제작

지난 시간까지, String이나 Int32와 같은 ROS2 기본 데이터형만을 메세지로 사용했다. 
코드로 예를 들면 이런 것이었다. 그간 만들었던 노드들의 코드 헤더를 보면 이렇게 되어있었을 것이다. 

```python
from std_msgs.msg import String
from std_msgs.msg import Int32
```

실제로 보내는 데이터들은 이런 식이었다. 

```python
msg = String()
msg.data = "hello"
```

하지만 문자열, int형 말고도 로봇 만들다보면 많은 데이터들이 들어가야 한다. 로봇 이름이나, 동작, 위치, 배터리 상태, 온도, 속도 등등등. 그걸 모두 메세지로 하게 되면 토픽이 4개나 되는데, 그건 매우 비효율적이다. 기본적으로 네트워크 통신방식을 사용하는 ROS의 특징상 데이터 패킷 앞뒤로 헤더같은것들이 붙어야 하기때문에, 데이터를 모두 구분해서 내보낸다는건 헤더들의 숫자가 늘어난다는 것이 된다. 

그래서 모바일 로봇의 베이스가 있다면, 그에 해당하는 컨트롤 데이터는 하나의 메세지로 묶는다거나 하는 방식을 사용한다. 

예를 들자면, `RobotStatus.msg` 라는 메세지를 다음과 같이 정의할 수 있다. 

```python
string name
float32 battery
float32 temperature
float32 velocity
float32 rotation
```

그러면 topic이 되었건 서비스 request가 되었건 한번에 묶어서 다음과 같이 보낼 수 있다. 
```python 
msg.name
msg.battery
msg.temperature
msg.velocity
msg.rotation
```

그럼 이 커스텀 메세지를 어떻게 보내는지 해보자. 

## 메세지 작성

이제 파일을 생성해보자. RobotStatus.msg 라는 텍스트 파일을 만들 예정이다.   
이걸 만들어 빌드를 하면, 노드 코드에서 import가 가능해진다. 

```python
from chapter6.msg import RobotStatus
```
이게 된다는 거다. 

일단 새로운 패키지를 하나 만들어보자. 

```bash
cd ~/ros2_ws/src
ros2 pkg create chapter6_msg
```

<figure>
  <img src="/assets/images/2026-06-23-21-56-49.png" style="width:80% !important; height:auto;" alt="2026-06-23-21-56-49">
  <figcaption>2026-06-23-21-56-49</figcaption>
</figure>

여기에 msg 폴더를 하나 만들고 만들려고 했던 메세지 파일도 만들어보자. 

```bash
cd chapter6_msg
mkdir msg
touch msg/RobotStatus.msg
```
이 RobotStatus.msg 파일에 다음과 같이 채워넣자. 

```pyhthon
string name
float32 battery
float32 temperature
float32 velocity
float32 rotation
```

## CMakeLists.txt 수정
빌드에 있어 중요한 과정이다.
다음 두 줄이 들어가야 한다. 
```
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotStatus.msg"
)
```


<figure>
  <img src="/assets/images/2026-06-23-22-47-37.png" style="width:80% !important; height:auto;" alt="2026-06-23-22-47-37">
  <figcaption>2026-06-23-22-47-37</figcaption>
</figure>

## package.xml 수정
빌드과정에서 생성기, 메세지사용등의 필요로 다음 세줄이 들어가야 한다. 

```xml
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

<figure>
  <img src="/assets/images/2026-06-23-22-50-54.png" style="width:80% !important; height:auto;" alt="2026-06-23-22-50-54">
  <figcaption>2026-06-23-22-50-54</figcaption>
</figure>

## 빌드 후 확인

이제 늘상 하는 것들을 해보자. colcon build, source....

<figure>
  <img src="/assets/images/2026-06-23-22-52-52.png" style="width:80% !important; height:auto;" alt="2026-06-23-22-52-52">
  <figcaption>2026-06-23-22-52-52</figcaption>
</figure>

여기에 추가로, 커스텀 메세지 형식이 만들어졌는지 확인해보자. 

명령어는 `ros2 interface show chapter6_msg/msg/RobotStatus`를 쓰면 된다. 

<figure>
  <img src="/assets/images/2026-06-23-22-53-58.png" style="width:80% !important; height:auto;" alt="2026-06-23-22-53-58">
  <figcaption>2026-06-23-22-53-58</figcaption>
</figure>

이제 다른 노드에서 이 메세지 타입을 사용할 수 있게 된다. 
빈 챕터였던 chapter5에서 한번 확인해보자. 

## chapter5에서. 

이제 비어있던  chapter5에 내용을 채워보자. 
노드 명세서를 작성하는 습관을 들여놓자. 

| 항목             | 내용                             |
| -------------- | ------------------------------ |
| 패키지 이름         | `chapter5`                     |
| Python 파일      | `robot_status_pub.py`          |
| Node 이름        | `robot_status_pub`             |
| Topic 이름       | `/robot_status`                |
| Message 타입     | `chapter6_msg/msg/RobotStatus` |
| Timer 주기       | `1.0 sec`                      |
| Timer Callback | `timer_callback()`             |




좀전에 package.xml을 만들고 분석만 했다. 
여기에 `<depend>` 를 넣어보자.

사실 이게 없어도 빌드가 될 수는 있다. 먼저 chapter6_msg가 빌드가 되어있다면,   
chapter5의 변경사항은 그러려니 하고 넘어갈 수도 있다.  
하지만, 코드가 배포되었을때 생기는 문제점들을 사전에 차단하기 위해 depend작업은 넣는게 정석이다. 

```xml
<depend>rclpy</depend>
<depend>chapter6_msg</depend>
```

<figure>
  <img src="/assets/images/2026-06-23-23-49-22.png" style="width:80% !important; height:auto;" alt="2026-06-23-23-49-22">
  <figcaption>2026-06-23-23-49-22</figcaption>
</figure>

이제 setup.py에 엔트리 포인트도 하나 넣어보자. 
```python
entry_points={
    'console_scripts': [
        'robot_status_pub = chapter5.robot_status_pub:main',
    ],
},
```

<figure>
  <img src="/assets/images/2026-06-23-23-54-15.png" style="width:80% !important; height:auto;" alt="2026-06-23-23-54-15">
  <figcaption>2026-06-23-23-54-15</figcaption>
</figure>


이제 노드 파일인 robot_status_pub.py를 작성해보자. 

```python
import rclpy
from rclpy.node import Node
from chapter6_msg.msg import RobotStatus

class RobotStatusPublisher(Node):

    def __init__(self):
        super().__init__('robot_status_publisher')

        self.publisher = self.create_publisher(
            RobotStatus,
            'robot_status',
            10
        )

        self.timer = self.create_timer(
            1.0,
            self.timer_callback
        )

    def timer_callback(self):
        msg = RobotStatus()

        msg.name = 'robot_1'
        msg.battery = 95.0
        msg.temperature = 36.5
        msg.velocity = 0.3

        self.publisher.publish(msg)

        self.get_logger().info(
            f'name: {msg.name}, '
            f'battery: {msg.battery}, '
            f'temperature: {msg.temperature}, '
            f'velocity: {msg.velocity}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = RobotStatusPublisher()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
```
헤더부분에 chapter6내용이 들어간게 눈에 띈다. 

<figure>
  <img src="/assets/images/2026-06-23-23-56-44.png" style="width:80% !important; height:auto;" alt="2026-06-23-23-56-44">
  <figcaption>2026-06-23-23-56-44</figcaption>
</figure>

만들고 빌드해본다. 

<figure>
  <img src="/assets/images/2026-06-23-23-57-34.png" style="width:80% !important; height:auto;" alt="2026-06-23-23-57-34">
  <figcaption>2026-06-23-23-57-34</figcaption>
</figure>

아.... 파일명에 오타가 있었다. 수정하고 다시....
setup.py를 수정하고 저장도 안했었네. 수정하고 다시...

<figure>
  <img src="/assets/images/2026-06-24-00-03-08.png" style="width:80% !important; height:auto;" alt="2026-06-24-00-03-08">
  <figcaption>2026-06-24-00-03-08</figcaption>
</figure>