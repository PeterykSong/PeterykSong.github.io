---
#order: 70
layout: single
title: "fast G-ICP tutorial."
date: 2025-08-14 21:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900

excerpt: "Fast G-ICP tutorial"
tags:
  - SLAM
  - Robotics  
  - ICP
---

학습을 위한 페이지입니다. 오류 사항이나 지적해야 할 부분 있으면 편하게 연락주세요. 


# Fast G-ICP
 - https://ieeexplore.ieee.org/document/9560835
 
  사용하고자 하는 코드 git은 다음과 같다. 

  https://github.com/koide3/fast_gicp


# 설치
  
  Chatgpt의 가이드를 충실히 따라 보았다.   
  쓰고있는 노트북의 GPU가 MX450이어서 CUDA쪽 문제가 있었다. CUDA를 끄기로 하고 컴파일하니 문제없이 컴파일이 된다. 아마, CUDA버전과 Toolkit쪽 문제인듯 싶다. 이 부분은 추후 임베디드 보드 사용하면서 다시 보도록 하자. 

## 1. 필수 패키지의 설치

```bash
sudo apt update
sudo apt install -y build-essential cmake git libeigen3-dev libomp-dev libpcl-dev
```

- libpcl-dev → PCL(Point Cloud Library) 전체 개발 헤더/라이브러리.
- libeigen3-dev → fast_gicp 내부 수학 연산에 필수.
- libomp-dev → 멀티스레드 지원.
- CUDA 사용 계획 있으면 드라이버와 CUDA Toolkit도 설치.

## 2. Build

#### CPU Only
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

```

#### GPU+CUDA

CUDA 툴킷을 설치해야 한다. 
내 경우앤 이 부분에서 빌드 오류가 나는 경우가 많아 CPU Onlu로 할 수밖에 없었다. 
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_VGICP_CUDA=ON
make -j$(nproc)
```


## 3. Binding python
바인딩이라는 용어는 C++로 빌드된 소스를 python에서 사용할 수 있도록 해주는걸 말한다. 용어의 설명이 부적절하겠지만 지금은 일단 그렇게 알아두자. 

```bash
cd fast_gicp
python3 setup.py install --user
```

이걸 하면 이제 pyhon에서 pygicp라는 모듈로 import가 가능해진다. 

```python
import numpy as np, pygicp
target = np.random.randn(1000,3).astype(np.float32)
source = target + np.array([0.5,0.2,0.1], np.float32)
T = pygicp.align_points(target, source, method="GICP", num_threads=8)
print(T)

```

바인딩을 시도 했더니 밑의 오류가 나온다. 

```bash
Traceback (most recent call last):
  File "/home/peteryksong/Documents/GitHub/fast_gicp/setup.py", line 7, in <module>
    from setuptools import setup, Extension
ModuleNotFoundError: No module named 'setuptools'
```

이건 Python에서 바인딩을 위한 모듈이 설치되어있지 않아 나오는 문제다. 

```bash
python3 -m pip install --upgrade pip
python3 -m pip install setuptools wheel pybind11 cmake numpy

```

추가적으로 아래의 오류가 나왔는데, CMAKE error가 뜬다. 
이 부분도 ChatGPT를 통해 물어보니 아래와 같은 해결책을 준다. 

```bash
-- Could NOT find pybind11 (missing: pybind11_DIR) CMake Error at CMakeLists.txt:95 (pybind11_add_module): Unknown CMake command "pybind11_add_module".
```
패키지를 추가로 설치하고, 빌드를 처음부터 다시 해본다.

```bash
sudo apt update
sudo apt install -y pybind11-dev python3-pybind1
```

이후 제일 앞부분의 바인딩을 다시 시도하면 pygicp 0.0.1이 만들어짐을 알 수 있다. 

```bash
Installed /home/peteryksong/.local/lib/python3.10/site-packages/pygicp-0.0.1-py3.10-linux-x86_64.egg
Processing dependencies for pygicp==0.0.1
Finished processing dependencies for pygicp==0.0.1

```


# 실행
여차저차 하여 위까지 설치를 마치면 한번 실행해보자. 

```bash
/fast_gicp$ cd data/
/fast_gicp/data$ ../build/gicp_align 251370668.pcd 251371071.pcd
```

그러면 아래와 같이 결과가 나오는 것을 볼 수 있다. 

```
target:17047[pts] source:17334[pts]
--- pcl_gicp ---
single:160.657[msec] 100times:15628.7[msec] fitness_score:0.204892
--- pcl_ndt ---
single:56.9838[msec] 100times:5891.56[msec] fitness_score:0.229377
--- fgicp_st ---
single:142.63[msec] 100times:14115.3[msec] 100times_reuse:9423.08[msec] fitness_score:0.204376
--- fgicp_mt ---
single:51.3404[msec] 100times:5054.43[msec] 100times_reuse:1790.87[msec] fitness_score:0.204384
--- vgicp_st ---
single:110.478[msec] 100times:10554.3[msec] 100times_reuse:5772.06[msec] fitness_score:0.205022
--- vgicp_mt ---
single:25.4905[msec] 100times:2390.74[msec] 100times_reuse:1391.59[msec] fitness_score:0.205022
```

- `target:17047[pts] source:17334[pts]` : 대상 포인트 수와 입력 포인트. 다운샘플링 후 프레임당 점의 개수를 말한다. 
- `single: [msec]` : 단일 정합 수행 시간
- `100times: [msec]` : 동일한 데이터를 100번 정합했을때 걸린 시간
- `100times_reuse: [msec]` : 100번 정합하되, KD-tree나 voxel grid 같은 전처리리를 할 경우 걸린 시간
- `fitness score` : 정합된 포인트들의 평균 제곱 오차

- pcl_gicp : PCL라이브러리의 기본 G-ICP (single thread)
- pcl_ndt : pcl 라이브러리의 NDT ICP
- fgicp_st : Fast-gicp의 Single Thread 버전
- fgicp_mt : Fast-gicp의 Multi Thread 버전
- vgicp_st : Voxelized Fast gicp의 single Thread버전
- vgicp_mt : Voxelized Fast gicp의 multi Thread버전


# 데이터셋 실험

이건 자료를 좀 모아서 정리해보면 좋을 것 같다.   
우선 임베디드 시스템으로 국한지어서, 여러 오픈된 알고리즘들을 임베디드 보드에서 실행시 정합이 얼마나 잘 되는가에 대한 통계를 내어보면, 3~4장짜리 보고서 논문정도는 가능하지 않을까 싶다. 천천히 한번 해보자. 

우선, Replica 데이터셋을 사용해보기로 한다. 뭐 워낙에 유명한 데이터셋이니까. 되겠지. 

```bash
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
./download.sh ~/data/replica_v1

```

