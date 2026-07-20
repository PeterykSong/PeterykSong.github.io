---
#order: 70
layout: single

title: "Image Filters"
date: 2026-06-15 00:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
related: false
topic : pattern_recognition
excerpt: "ImageFilters"
tags:
  - Robotics
  - Pattern Recognition
  - Image Filters
  
---

논문을 읽으면서 나오는 이미지 필터/기하 추출방법들 실습칸입니다. 

{: .notice}

# 들어가면서.   

논문을 읽다보면 여러가지 형태들의 Filter들이 언급된다.   
잘 모를땐 그러려니 하고 넘기는데, 점점 나도 한번쯤은 해봐야지 하는 생각을 하게 된다.   
그래서 해보려 한다. 근데 뭐 이리 많냐..  
<br>


# Intro 

해볼것들의 목록이다.   Floorplan관련 논문을 읽다보니 나온것들이라서, 범용적인건 아닐수도 있지만. 

| 순서 | 필터 / 연산                                               | 등장 논문                                      | 적용 대상                                      | 역할                                                                                                             |
| -- | ----------------------------------------------------- | ------------------------------------------ | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| 1  | **Gaussian filter / Gaussian smoothing**              | **PALMS+**                                 | 관측 wall segment로 만든 layout-matching kernel | 관측 구조와 floorplan 사이의 작은 오차에 강건하게 만들기 위해 RW kernel을 Gaussian smoothing함.                                        |
| 2  | **Canny edge detection**                              | **PALMS+**                                 | 3D point cloud를 수평면에 투영한 2D raster image   | 벽·문틀 같은 구조 경계를 추출하기 위해 사용. 이후 Hough transform으로 line segment화함.                                                |
| 3  | **Probabilistic Hough Transform**                     | **PALMS+**                                 | Canny edge 결과                              | dominant structural boundary를 line segment로 변환.                                                                |
| 4  | **Binarization**                                      | **SignLoc**                                | venue map / floor plan                     | 컬러, 텍스트, 심볼 등 clutter를 제거하고 geometry를 담은 binary image로 변환. Fig. 2에도 map extraction pipeline으로 표시됨.             |
| 5  | **Polygonal skeletonization**                         | **SignLoc**                                | traversable area의 binary map               | 이동 가능한 영역의 중심선을 추출해 navigation graph를 만드는 데 사용.                                                                |
| 6  | **Text / symbol extraction**                          | **SignLoc**                                | floor plan / venue map                     | OCR, VLM 등을 이용해 텍스트와 심볼을 추출하고 graph node label이나 portal node 구성에 사용. 엄밀한 이미지 필터라기보다는 map parsing 단계.           |
| 7  | **Mask2Former semantic segmentation**                 | **Z-FLoc**                                 | RGB image                                  | wall/floor 영역을 분할하고 가구 같은 clutter를 억제해 wall-only 3D points를 얻음.                                                |
| 8  | **Wall mask filtering / column-wise depth filtering** | **Z-FLoc**                                 | gravity-aligned wall mask + depth map      | 각 이미지 column의 대표 wall depth를 median으로 계산하고, 비벽 픽셀이 더 깊게 나타나는 column을 제거하여 ghost wall artifact를 줄임.             |
| 9  | **Median depth filtering**                            | **Z-FLoc**                                 | column-wise wall depth                     | column별 대표 벽 깊이를 median으로 잡아 잘못된 wall mask를 정제.                                                                |
| 10 | **Seed-based region growing**                         | **Z-FLoc**                                 | BEV wall pixels                            | BEV에서 line segment를 추출하기 위한 region growing 기반 line extraction.                                                 |
| 11 | **PCA-based dominant direction estimation**           | **Z-FLoc**                                 | BEV wall pixel neighborhood                | line model 초기화를 위해 local neighborhood의 dominant direction을 추정.                                                 |
| 12 | **HDBSCAN clustering**                                | **Z-FLoc**                                 | BEV wall pixels                            | 원형 구조, 예를 들어 기둥 등을 찾기 위해 wall pixel cluster를 분리.                                                               |
| 13 | **RANSAC circle fitting**                             | **Z-FLoc**                                 | HDBSCAN cluster                            | line 외에 circle primitive를 추출하기 위해 3-point RANSAC으로 원을 fitting.                                                 |
| 14 | **Line deduplication / near-duplicate suppression**   | **Z-FLoc**                                 | BEV / floorplan line primitives            | 방향과 offset이 비슷한 중복 line을 제거하고 더 긴 segment를 유지.                                                                 |
| 15 | **LSD, Line Segment Detector**                        | **Fully Geometric Panoramic Localization** | panorama에서 자른 perspective crop             | 2D line segment 검출에 사용.                                                                                        |
| 16 | **ELSED**                                             | **Fully Geometric Panoramic Localization** | line detector variation 실험                 | LSD 대신 비교한 line segment detector.                                                                              |
| 17 | **DeepLSD**                                           | **Fully Geometric Panoramic Localization** | line detector variation 실험                 | 딥러닝 기반 line segment detector로 비교 실험에 언급됨.                                                                      |
| 18 | **Line distance function / Point distance function**  | **Fully Geometric Panoramic Localization** | 2D/3D line 및 intersection 분포               | 선과 교점의 분포를 distance field로 표현해 pose search에 사용. 전통적 영상 필터는 아니지만, line image/geometric map에 대한 거리장 필터로 볼 수 있음.  |
| 19 | **Majority voting interpolation**                     | **Semantic Rays**                          | semantic ray vector                        | discrete semantic label ray를 줄일 때 linear interpolation 대신 window 내 다수결로 label을 선택.                             |
| 20 | **Semantic ray prediction / semantic label masking**  | **Semantic Rays**                          | RGB image + semantic floorplan             | ResNet50 기반 네트워크로 wall/window/door semantic ray를 예측하고 semantic probability volume 생성.                          |
| 21 | **Gravity-alignment warping**                         | **F3Loc, UnLoc, Z-FLoc**                   | 기울어진 RGB image / mask                      | roll-pitch 영향을 줄이기 위해 이미지를 중력 방향에 맞게 정렬. 엄밀히는 필터라기보다 geometric image warping.                                  |
| 22 | **Binary mask from gravity alignment**                | **UnLoc**                                  | gravity-aligned image feature              | gravity alignment 과정에서 생긴 valid region mask를 depth/uncertainty prediction에 사용.                                 |



와.... 이거 언제 다해보지..?