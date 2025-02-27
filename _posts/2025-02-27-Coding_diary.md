---
title: "2월 4주차 코딩연습일기"

#excerpt: "첫번째 포스트, Github Blog를 시작하면서"

last_modified_at: 2025-2-26 19:44:00 +0900
toc: false

header:
  #teaser: /assets/images/2025-01-09_post.jpg #첫페이지에서 요약으로 보이는 페이지.
  #overlay_image: /assets/images/2025-01-09_post.png #포스트 페이지에서 보이는 이미지
  #overlay_filter: 0.5 #숫자가 높을수록 어둡다.
  #overlay_image: ""
  #overlay_color : "rgb(255,255,255)"

tags:
  - Weekly  
---

Linked List를 배워보자. 
{: .notice}
# 문제 
  You are given two non-empty linked lists representing two non-negative integers. 
  The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

  You may assume the two numbers do not contain any leading zero, except the number 0 itself.

# 예제
  Input: l1 = [2,4,3], l2 = [5,6,4]
  Output: [7,0,8]
  Explanation: 342 + 465 = 807.

  Example 2:

  Input: l1 = [0], l2 = [0]
  Output: [0]

  Example  3:

  Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
  Output: [8,9,9,9,0,0,0,1]

일단 Linked List를 처음 겪어봤다. 
나중에 SLAM코드 짜면, Graph slam부분에서 응용되기 쉬울듯하다.
만약 Object SLAM에서 인식된 Object들 간의 관계를 정의하기도 좋을 것 같다.

해답지는 다음과 같다. 


```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    numberL1 = 0  # numberL1 초기화
    digit = 0
    while l1:  # l1이 None이 아닐 때까지 반복
        numberL1 += l1.val * (10 ** digit)  # l1.val 사용
        l1 = l1.next  # 다음 노드로 이동
        digit += 1
    
    numberL2 = 0  # numberL2 초기화
    digit = 0
    while l2:  # l2가 None이 아닐 때까지 반복
        numberL2 += l2.val * (10 ** digit)  # l2.val 사용
        l2 = l2.next  # 다음 노드로 이동
        digit += 1

    L3 = numberL1 + numberL2

    # 결과를 ListNode 형태로 변환
    dummy_head = ListNode(0)  # 더미 헤드 노드
    current = dummy_head

    if L3 == 0:  # L3가 0일 경우
        return ListNode(0)

    while L3 > 0:  # L3가 0보다 큰 경우에만 반복
        current.next = ListNode(L3 % 10)  # 새로운 노드 생성
        current = current.next  # 현재 노드 이동
        L3 = L3 // 10  # 정수 나누기로 수정
    
    return dummy_head.next  # 더미 헤드의 다음 노드를 반환
```

기왕 하는김에, C++로 구현하면 아래처럼 된다. 

```cpp
#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        long long numberL1 = 0;  // numberL1 초기화
        long long numberL2 = 0;  // numberL2 초기화
        int digit = 0;

        while (l1) {  // l1이 nullptr이 아닐 때까지 반복
            numberL1 += l1->val * pow(10, digit);  // l1->val 사용
            l1 = l1->next;  // 다음 노드로 이동
            digit++;
        }

        digit = 0;
        while (l2) {  // l2가 nullptr이 아닐 때까지 반복
            numberL2 += l2->val * pow(10, digit);  // l2->val 사용
            l2 = l2->next;  // 다음 노드로 이동
            digit++;
        }

        long long L3 = numberL1 + numberL2;

        // 결과를 ListNode 형태로 변환
        ListNode* dummyHead = new ListNode(0);  // 더미 헤드 노드
        ListNode* current = dummyHead;

        if (L3 == 0) {  // L3가 0일 경우
            return new ListNode(0);
        }

        while (L3 > 0) {  // L3가 0보다 큰 경우에만 반복
            current->next = new ListNode(L3 % 10);  // 새로운 노드 생성
            current = current->next;  // 현재 노드 이동
            L3 /= 10;  // 정수 나누기로 수정
        }

        return dummyHead->next;  // 더미 헤드의 다음 노드를 반환
    }
};
```

