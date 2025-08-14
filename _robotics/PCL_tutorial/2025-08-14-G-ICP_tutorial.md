---
#order: 70
#layout: post
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

  일단 심플 예제는 되었고, 이제 데이터셋을 넣고 한번 돌려보는게 목적이다. 이번주에는 어떻게든 되겠지. 