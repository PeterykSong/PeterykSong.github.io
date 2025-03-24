import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# DH 파라미터 정의 (theta, d, a, alpha)
dh_params = [
    [0, 0, 1, 0],      # 링크 1: 회전 조인트
    [np.pi/2, 0, 1, 0], # 링크 2: 유니버설 조인트
    [0, 0.5, 0, 0],    # 링크 3: 프리즘 조인트
    [0, 0, 1, 0],      # 링크 4: 회전 조인트
    [0, 0, 1, np.pi/2], # 링크 5: 스크류 조인트
    [0, 0, 1, 0]       # 링크 6: 회전 조인트
]

# 조인트 종류에 따른 마커와 색상 정의
joint_types = ['R', 'U', 'P', 'R', 'S', 'R']  # R: 회전, U: 유니버설, P: 프리즘, S: 스크류
markers = ['o', 's', 'D', '^', 'v', 'x']  # 각 조인트에 대한 마커
colors = ['b', 'g', 'r', 'c', 'm', 'y']  # 각 조인트에 대한 색상

def dh_transform(theta, d, a, alpha):
    """DH 변환 행렬을 계산합니다."""
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(dh_params):
    """DH 파라미터를 사용하여 순방향 운동학을 계산합니다."""
    T = np.eye(4)  # 초기 변환 행렬
    points = [[0, 0, 0]]  # 시작점

    for params in dh_params:
        theta, d, a, alpha = params
        T = T @ dh_transform(theta, d, a, alpha)  # 변환 행렬 업데이트
        points.append(T[:3, 3])  # 현재 위치 저장

    return np.array(points)

# 링크 플로팅
points = forward_kinematics(dh_params)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, point in enumerate(points):
    # 인덱스 오류를 방지하기 위해 markers와 colors의 길이를 확인
    if i < len(markers) and i < len(colors):
        ax.scatter(point[0], point[1], point[2], marker=markers[i], color=colors[i], label=f'Joint {i+1} ({joint_types[i]})')
        ax.text(point[0], point[1], point[2], str(i+1), fontsize=12, ha='right')  # 조인트 번호 표시
        
        # X, Y, Z 방향으로 벡터 그리기
        ax.quiver(point[0], point[1], point[2], 1, 0, 0, color='r', arrow_length_ratio=0.1)  # X 방향 벡터
        ax.quiver(point[0], point[1], point[2], 0, 1, 0, color='g', arrow_length_ratio=0.1)  # Y 방향 벡터
        ax.quiver(point[0], point[1], point[2], 0, 0, 1, color='b', arrow_length_ratio=0.1)  # Z 방향 벡터

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_title('3D DH Parameter Link Plot with Joint Types and Vectors')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid()
ax.legend()
plt.show() 