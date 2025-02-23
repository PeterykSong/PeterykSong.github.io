import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

# 랜덤 시드 설정
np.random.seed(42)

# 환경 설정
num_particles = 100  # 파티클 개수
num_landmarks = 5  # 랜드마크 개수
timesteps = 20  # 로봇 이동 단계

# 랜드마크 실제 위치 (고정)
landmarks = np.array([[5, 10], [15, 5], [10, 15], [20, 10], [10, 5]])

# 로봇 초기 상태 (x, y, 방향)
true_robot_state = np.array([0.0, 0.0, 0.0])  

# 파티클 초기화 (x, y, 방향)
particles = np.array([true_robot_state + np.random.randn(3) * 0.5 for _ in range(num_particles)])
particle_weights = np.ones(num_particles) / num_particles  # 초기 가중치

# EKF용 랜드마크 상태 (각 랜드마크의 평균과 공분산 초기화)
landmark_estimates = {i: {"mean": np.random.randn(2) * 0.5 + landmarks[i], "cov": np.eye(2) * 0.5}
                      for i in range(num_landmarks)}

# 이동 모델 노이즈 (x, y, 방향) - 노이즈 감소하여 수렴 효과 강화
motion_noise = np.diag([0.2, 0.2, 0.05])  
sensor_noise = np.diag([0.5, 0.5])  # 센서 노이즈 감소

# 플롯 설정
fig, ax = plt.subplots(figsize=(8, 6))

# FastSLAM 2.0 실행 (입자가 점점 모이도록 개선)
for t in range(timesteps):
    # 로봇 이동 (전진)
    move = np.array([1.0, 0.5, 0.1])  # (전진, y 이동, 방향 회전)
    true_robot_state += move + np.random.multivariate_normal([0, 0, 0], motion_noise)
    
    # 입자 필터 업데이트 (입자 이동)
    for i in range(num_particles):
        particles[i, :3] += move + np.random.multivariate_normal([0, 0, 0], motion_noise * 0.5)

    # 랜드마크 관측 및 EKF 업데이트
    for i, landmark in enumerate(landmarks):
        observed_landmark = landmark + np.random.multivariate_normal([0, 0], sensor_noise)  # 랜드마크 측정값

        # EKF 업데이트
        for p in range(num_particles):
            ekf_mean = landmark_estimates[i]["mean"]
            ekf_cov = landmark_estimates[i]["cov"]

            # 관측 모델의 야코비안 행렬
            H = np.eye(2)
            S = H @ ekf_cov @ H.T + sensor_noise
            K = ekf_cov @ H.T @ np.linalg.inv(S)  # 칼만 이득

            # EKF 상태 업데이트
            innovation = observed_landmark - ekf_mean
            ekf_mean += K @ innovation
            ekf_cov = (np.eye(2) - K @ H) @ ekf_cov

            # 업데이트된 EKF 값 저장
            landmark_estimates[i]["mean"] = ekf_mean
            landmark_estimates[i]["cov"] = ekf_cov

    # 입자 가중치 업데이트 (입자가 점점 모이도록 강화)
    for i in range(num_particles):
        weight = 1.0
        for j, landmark in enumerate(landmarks):
            expected_landmark = landmark_estimates[j]["mean"]
            weight *= multivariate_normal.pdf(landmarks[j], mean=expected_landmark, cov=sensor_noise * 0.5)
        particle_weights[i] = weight

    # 가중치 정규화
    particle_weights += 1.e-300  # 작은 값 추가하여 0 방지
    particle_weights /= np.sum(particle_weights)

    # ✅ 리샘플링을 더 강하게 수행 (입자들이 점점 모이도록 개선)
    N_eff = 1.0 / np.sum(np.square(particle_weights))  # 유효 샘플 개수 계산
    if N_eff < num_particles / 2:  # 유효 샘플 수가 절반 이하로 감소하면 리샘플링
        indices = np.random.choice(range(num_particles), size=num_particles, p=particle_weights)
        particles = particles[indices]
        particle_weights = np.ones(num_particles) / num_particles  # 리샘플링 후 균등 가중치 재설정

    # 시각화 업데이트
    ax.clear()
    ax.set_xlim(-5, 25)
    ax.set_ylim(-5, 20)

    # 실제 로봇 경로
    ax.plot(true_robot_state[0], true_robot_state[1], 'bo', label="True Robot Position")

    # 입자 표시 (점점 수렴하는 형태)
    ax.scatter(particles[:, 0], particles[:, 1], c='r', s=2, alpha=0.5, label="Particles")

    # 실제 랜드마크
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='g', marker='x', s=100, label="True Landmarks")

    # FastSLAM 2.0이 추정한 랜드마크
    estimated_landmarks = np.array([landmark_estimates[i]["mean"] for i in range(num_landmarks)])
    ax.scatter(estimated_landmarks[:, 0], estimated_landmarks[:, 1], c='orange', marker='o', label="Estimated Landmarks")

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"FastSLAM 2.0 (Iteration {t+1})")
    plt.grid()

    # 1초 대기 후 업데이트
    plt.pause(1.0)

plt.show()
