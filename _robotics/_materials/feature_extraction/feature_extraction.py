import cv2
import matplotlib.pyplot as plt
import time  # 시간 측정을 위한 모듈 추가


# 이미지 로드
img = cv2.imread('target.png', cv2.IMREAD_GRAYSCALE)

# SIFT 특징 추출
start_time = time.time()  # 시작 시간 기록
sift = cv2.SIFT_create()
keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)
sift_time = time.time() - start_time  # 소요 시간 계산
print(f"SIFT 특징 개수: {len(keypoints_sift)}, 계산 시간: {sift_time:.4f}초")
print("SIFT Descriptors:")
print(descriptors_sift)  # SIFT descriptor 출력

# ORB 특징 추출
start_time = time.time()  # 시작 시간 기록
orb = cv2.ORB_create()
keypoints_orb, descriptors_orb = orb.detectAndCompute(img, None)
orb_time = time.time() - start_time  # 소요 시간 계산
print(f"ORB 특징 개수: {len(keypoints_orb)}, 계산 시간: {orb_time:.4f}초")
print("ORB Descriptors:")
print(descriptors_orb)  # ORB descriptor 출력

# A-KAZE 특징 추출
start_time = time.time()  # 시작 시간 기록
akaze = cv2.AKAZE_create()
keypoints_akaze, descriptors_akaze = akaze.detectAndCompute(img, None)
akaze_time = time.time() - start_time  # 소요 시간 계산
print(f"A-KAZE 특징 개수: {len(keypoints_akaze)}, 계산 시간: {akaze_time:.4f}초")
print("A-KAZE Descriptors:")
print(descriptors_akaze)  # A-KAZE descriptor 출력

# 특징 시각화 (점으로 표현)
img_sift = cv2.drawKeypoints(img, keypoints_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_orb = cv2.drawKeypoints(img, keypoints_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_akaze = cv2.drawKeypoints(img, keypoints_akaze, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Descriptor의 처음 10개를 큰 점으로 표현
for i in range(min(10, len(keypoints_sift))):
    x, y = int(keypoints_sift[i].pt[0]), int(keypoints_sift[i].pt[1])
    cv2.circle(img_sift, (x, y), 9, (0, 255, 0), -1)  # SIFT descriptor 점 크기 증가

for i in range(min(10, len(keypoints_orb))):
    x, y = int(keypoints_orb[i].pt[0]), int(keypoints_orb[i].pt[1])
    cv2.circle(img_orb, (x, y), 9, (0, 255, 0), -1)  # ORB descriptor 점 크기 증가

for i in range(min(10, len(keypoints_akaze))):
    x, y = int(keypoints_akaze[i].pt[0]), int(keypoints_akaze[i].pt[1])
    cv2.circle(img_akaze, (x, y), 9, (0, 255, 0), -1)  # A-KAZE descriptor 점 크기 증가

# 결과 출력
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_sift, cmap='gray')
plt.title('SIFT Keypoints with Descriptors')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_orb, cmap='gray')
plt.title('ORB Keypoints with Descriptors')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_akaze, cmap='gray')
plt.title('A-KAZE Keypoints with Descriptors')
plt.axis('off')

plt.show()
