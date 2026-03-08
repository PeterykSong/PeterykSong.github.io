---
#order: 70
layout: single

title: "Simulator"
date: 2026-03-08 00:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900
related: false

excerpt: "Day 2"
tags:
  - Robotics
  - KiCAD
  - Hardware
  
---

회사생활 15년동안 했었던 Hardware 이야기를 해보려 합니다. 
{: .notice}

# 설계란......   

신입이 들어와 회로설계를 맡겼다.   
진도체크를 하기 위해 어디까지 했냐 했더니, 설계는 다 했고, 이제 저항이나 전압값들 맞춰보고 있단다.  
그 말을 들은 나와 선배의 말.  
>아직 안 그렸네!  

<br>
그렇다. 회로도처럼 그린 것은 그저 그림 일 뿐. 설계는 숫자가 채워져야 완성이 된다. 
<br>
<br>

# 숫자 채워넣기

회로도를 그리다보면 채워야 할 숫자가 참 많다.   

>각종 device들에 맞는 전압  
>전압을 만들기 위한 전원회로    
>전원회로를 구성하는 feedback 설정 저항  
>FET 설계는 제대로 된걸까  
>TR의 Base전류는 잘 설계된걸까?   
>전압의 변동에 따른 동작은 잘 될까?   

이런 숫자들을 채워넣을때, 가장 흔하디 흔하게 보는 것은 Datasheet들이다.   
제조사에서 권장하는 회로도를 그리고, 입력과 출력에 맞춰 내가 원하는 동작을 할 수 있도록 하나하나 계산을 해보며 숫자를 채워 넣는다. 

<figure>
  <img src="/assets/images/2026-03-08-20-50-14.png" style="max-width:none;">
  <figcaption>저기에 있는 R, L, C들을 채워 넣어야 한다. </figcaption>
</figure>

<figure>
  <img src="/assets/images/2026-03-08-20-51-06.png" style="max-width:none;">
  <figcaption>데이터시트에 있는 계산식을 다 따라가다보면 이런 숫자의 결과에 도달한다. </figcaption>
</figure>

방금 채워진 숫자에서 R_FBT와 R_FBB를 보면, 아주 괴랄한 특수 저항을 쓰고 있다.   
뭐, 100만개 정도 찍어낸다면야 저런 특수저항 두개즈음 늘어나는건 일도 아니겠지만.   
몇 천개 찍어내는데 저런 특수저항까지 챙기려고 하면 점점 빡침이 늘어날 뿐이다.

그때 제일 먼저 시도해보는건 가장 적당한 비율을 찾아내는거다.   

이제 표준 저항값들을 찾아보자. 
자료는 땜스전자연구소 블로그에서 가져왔다. 참고할게 많은 곳이다. 
https://blog.naver.com/ansdbtls4067/221329755138

<figure>
  <img src="/assets/images/2026-03-08-20-58-17.png" style="max-width:none;">
  <figcaption>표준저항값 찾기</figcaption>
</figure>

이제 가장 원초적인 시뮬레이터를 실행해본다.

그렇다. 스프레드시트 프로그램. 이름하여 **Excel**이다.   
여러 값들을 하나하나 넣어보면서, 가장 적절한 비율이 얼마인가를 찾아내본다. 
데이터시트의 수식을 조금만 더 찾아보면, Feedback 전압값에 따라 나오는 출력수식도 찾을 수 있으니,   
그것도 엑셀시트에 넣으면 한번에 계산이 가능하다. 

<figure>
  <img src="/assets/images/2026-03-08-21-01-41.png" style="max-width:none;">
  <figcaption>나는 지금 리눅스라서, Libre를 썼다. </figcaption>
</figure>

<figure>
  <img src="/assets/images/2026-03-08-21-02-32.png" style="max-width:none;">
  <figcaption>요샌 수식 유도하기가 잘 안된다.</figcaption>
</figure>

회로설계의 많은 것은 이렇게 엑셀로 할 수 있지만,   
그런다고 모든 것을 엑셀로 할 순 없다. 특히 Time-variant 환경에서 본다거나, 주파수 분석을 해보기 위해선 반드시 시뮬레이션을 돌려야 하는 환경이 등장한다. 그게 바로 SPICE 다. 

# SPICE

SPICE는 **S**imulation **P**rogram with **I**ntergrarted **C**ircuit **E**mphasis의 약자다. 자세한 설명은 ROHM 홈페이지에 잘 나와있다. ROHM도 이런 실무적인 자료를 참 잘 올려주는 곳이다. 여러모로 참고할게 많다.   
https://techweb.rohm.co.kr/product/circuit-design/simulation/7688/

예전엔 OrCAD 의 SPICE가 좋아서 보통 그걸 활용하라고 배웠는데, 요새는 왠만한 SPICE툴들의 정확도가 상향평준화가 되어서 익숙한걸 쓰면 좋다 수준이 되었다.   

그래서 회사 다니던 시절엔 **Circuit Labs**라는 툴을 즐겨썼었고 주변 동료들에게 널리 추천해서, 적어도 내가 있던 팀은 나에게 영향을 받아 공용아이디 만들고 유료결제 해서 돌려쓰게 했다.   

지금도 난 내 개인 결제로 쓰고 있다. (달달이 16달러인데, 그동안의 자료가 있어서 추억을 유지하기 위한 비용지출이 되어버렸다.)  
https://www.circuitlab.com/

<figure>
  <img src="/assets/images/2026-03-08-21-10-03.png" style="max-width:none;">
  <figcaption>Circuit Labs. 유료결제하면 저장소가 무한이다. </figcaption>
</figure>

<figure>
  <img src="/assets/images/2026-03-08-21-11-01.png" style="max-width:none;">
  <figcaption>마지막 업데이트가 2024년이네.. </figcaption>
</figure>

# KiCAD

KiCAD 도 시뮬레이션 툴을 지원한다. 
이전 버전에서는 시뮬레이션 툴이 프로젝트 화면에 있었는데, 지금은 회로도 편집기 안에 시뮬레이션 툴이 들어가있다. 

<figure>
  <img src="/assets/images/2026-03-08-21-15-15.png" style="max-width:none;">
  <figcaption>프로젝트 화면엔 없다.</figcaption>
</figure>

<figure>
  <img src="/assets/images/2026-03-08-21-19-17.png" style="max-width:none;">
  <figcaption>요기로 들어가있다.</figcaption>
</figure>

툴의 사용법은 다른 인터넷 찾아보면 많이 있다. 

제일 먼저 해야 할건, 전원과 GND의 설정.   
OrCAD가 생긴 이래, GND에는 반드시 시뮬레이션 그라운드를 붙여줘야 한다. 
<figure>
  <img src="/assets/images/2026-03-08-21-20-47.png" style="max-width:none;">
  <figcaption>2026-03-08-21-20-47</figcaption>
</figure>

다음 전원 역시 simulation 전용 전원을 붙여 줘야 한다. 

<figure>
  <img src="/assets/images/2026-03-08-21-22-30.png" style="max-width:none;">
  <figcaption>SPICE로 검색하면 쉽게 찾을 수 있다. </figcaption>
</figure>


VDC 심벌을 찾아 넣고 심벌 속성을 편집하면, 이렇게 5V가 되어있는걸 볼 수 있다. 

<figure>
  <img src="/assets/images/2026-03-08-21-24-35.png" style="max-width:none;">
  <figcaption>VDC를 찾아보자</figcaption>
</figure>

<figure>
  <img src="/assets/images/2026-03-08-21-23-47.png" style="max-width:none;">
  <figcaption>심벌을 더블클릭하면 뜬다.</figcaption>
</figure>

<figure>
  <img src="/assets/images/2026-03-08-21-24-17.png" style="max-width:none;">
  <figcaption>5V가 반영되었다. </figcaption>
</figure>

이제 좀전에 빨간 동그라미 첬던 시뮬레이터 버튼을 누르면 아래의 창이 뜨고,   
재생 버튼을 누르면 시뮬레이션이 실행된다. 

<figure>
  <img src="/assets/images/2026-03-08-21-26-01.png" style="max-width:none;">
  <figcaption>시뮬레이션 설정창</figcaption>
</figure>

조금 신경써서 볼만한건 분석유형인데, 지금은 제일 위의 항목인 DC동작점을 눌러 결과를 보자.
<figure>
  <img src="/assets/images/2026-03-08-21-26-51.png" style="max-width:none;">
  <figcaption>시뮬레이션 종류를 설정해야 한다.</figcaption>
</figure>

그러면 회로도에 전류값, 각 노드별 전압값이 뜬다. 
이제 가장 간단한 시뮬레이션을 해봤다. 

<figure>
  <img src="/assets/images/2026-03-08-21-28-22.png" style="max-width:none;">
  <figcaption>회로도에 간단한 결과값이 떠있다. DC Solver값이라고도 한다. </figcaption>
</figure>

이런식으로, 회로를 그렸으면 한번즘 동작 시뮬레이션을 했으면 한다.   
그게 손이 되었건, 엑셀이 되었건, SPICE 툴이건 중요하지 않다.  
숫자가 있는것과 없는 것은 천지차이다. 

이게 얼마나 달라지는지, 다음번엔 TR설계를 통해 알아보자. 