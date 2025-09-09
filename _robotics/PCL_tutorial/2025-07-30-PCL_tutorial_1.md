---
#order: 70
layout: single
title: "PCL_Tutorial (작성중)"
date: 2025-07-30 21:00:00 +0900
#last_modified_at: 2021-11-15 14:39:23 +0900

excerpt: "PCL Tutorial "
tags:
  - SLAM
  - RGB-D
  - Robotics  
  - ICP
---




# 3D Gaussian Splatting (3DGS) with NPU Implementation Guide

This document summarizes the implementation of 3D Gaussian Splatting (3DGS) using Neural Processing Units (NPUs) without CUDA, along with relevant research papers. The content is based on a conversation with Grok 3, created by xAI, on July 16, 2025.

---

## 1. Implementing 3D Gaussian Splatting on NPU without CUDA

### Overview
3D Gaussian Splatting (3DGS) is a high-performance rendering technique that relies heavily on parallel computing, typically optimized for GPUs using CUDA. Implementing it on NPUs, which are optimized for deep learning tasks, requires adapting the algorithm to leverage NPU strengths (e.g., matrix operations) while addressing limitations (e.g., graphics-specific tasks like rasterization).

### Key Steps and NPU Utilization
#### 1. Gaussian Definition and Projection
- **Task**: Define 3D Gaussians (position, covariance, color) and project them onto a 2D plane.
- **NPU Utilization**:
  - Matrix operations (covariance and projection) are well-suited for NPU matrix units (e.g., Apple AMX, Google TPU MXU).
  - Use frameworks like PyTorch (MPS backend) or TensorFlow (TPU backend).
  - **Example (PyTorch)**:
    ```python
    import torch
    device = torch.device("mps")  # Apple M1/M2 NPU
    positions = torch.tensor(gaussians_positions, device=device)
    cov_matrices = torch.tensor(gaussians_covariances, device=device)
    projection_matrix = torch.tensor(camera_projection, device=device)
    projected_gaussians = torch.matmul(positions, projection_matrix)
    ```

#### 2. Sorting
- **Task**: Sort projected Gaussians by depth for rendering order.
- **NPU Utilization**:
  - Sorting (e.g., Radix Sort) is less efficient on NPUs compared to GPUs.
  - Use built-in functions (e.g., `torch.sort`) or offload to CPU.
  - **Example**:
    ```python
    depths = torch.tensor(gaussians_depths, device=device)
    sorted_indices = torch.argsort(depths, descending=True)
    sorted_gaussians = projected_gaussians[sorted_indices]
    ```

#### 3. Rasterization
- **Task**: Render 2D Gaussians onto the screen with alpha blending.
- **NPU Utilization**:
  - Rasterization is GPU-centric; NPUs are less efficient.
  - Divide into tiles and convert to matrix operations for NPU processing.
  - **Example**:
    ```python
    for tile in tiles:
        tile_gaussians = filter_gaussians_in_tile(sorted_gaussians, tile)
        colors = torch.matmul(tile_gaussians, color_weights, device=device)
        alpha = torch.sigmoid(opacity_weights, device=device)
        rendered_tile = alpha_blending(colors, alpha)
    ```

#### 4. Optimization
- **Task**: Optimize Gaussian parameters (position, covariance, color) using gradient descent.
- **NPU Utilization**:
  - NPUs excel at gradient-based optimization.
  - Use PyTorch/TensorFlow autograd for loss computation and backpropagation.
  - **Example**:
    ```python
    optimizer = torch.optim.Adam(gaussians_parameters, lr=0.01)
    for epoch in range(num_epochs):
        rendered_image = render_gaussians(gaussians_parameters, device)
        loss = torch.nn.functional.mse_loss(rendered_image, target_image)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    ```

### NPU-Specific Considerations
- **Apple Neural Engine (M1/M2)**:
  - Use PyTorch MPS backend or Core ML.
  - Custom Metal shaders for rasterization if needed.
- **Google TPU**:
  - Use TensorFlow or JAX with XLA compiler.
  - Offload sorting/rasterization to CPU or custom ops.
- **Qualcomm Hexagon**:
  - Use Snapdragon NPE SDK or ONNX Runtime.
  - Optimize for mobile with lightweight scenes.

### Optimization Strategies
- Leverage NPU matrix operations for projection and optimization.
- Offload sorting and rasterization to CPU or custom NPU ops.
- Reduce memory usage (e.g., fewer Gaussians, smaller tiles) for NPU constraints.

### Open-Source Resources
- **gsplat**: PyTorch-based 3DGS library with MPS support (https://github.com/nerfstudio-project/gsplat).
- **graphdeco-inria/gaussian-splatting**: Reference CUDA implementation for porting.

---

## 2. Relevant Research Papers

### Overview
No papers directly address 3DGS implementation on NPUs, as the algorithm is optimized for CUDA-based GPUs. However, some studies indirectly suggest NPU applicability through MLP integration, compression, or lightweight architectures.

### Key Papers
1. **“3D Gaussian Splatting for Real-Time Radiance Field Rendering” (Kerbl et al., SIGGRAPH 2023)**:
   - Introduces 3DGS with CUDA-based optimization.
   - **NPU Relevance**: Matrix operations and optimization are NPU-compatible; sorting/rasterization needs CPU/custom ops.
   - **Source**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

2. **“NeuralGS: Bridging Neural Fields and 3D Gaussian Splatting for Compact 3D Representations” (2025)**:
   - Uses MLP to encode Gaussian attributes, reducing memory usage.
   - **NPU Relevance**: MLP operations are ideal for NPUs (e.g., Apple Neural Engine, Google TPU).
   - **Source**: Not publicly available; referenced in recent surveys.

3. **“A Survey on 3D Gaussian Splatting” (Chen et al., 2024)**:
   - Reviews 3DGS advancements, including optimization and hardware acceleration.
   - **NPU Relevance**: Suggests potential for NPU in optimization tasks.
   - **Source**: https://arxiv.org/abs/2402.12490

4. **“Recent Advances in 3D Gaussian Splatting” (Wu et al., 2024)**:
   - Discusses dynamic 3DGS and compression (e.g., LightGaussian).
   - **NPU Relevance**: MLP-based encoders and compression suit NPU’s memory constraints.
   - **Source**: https://arxiv.org/abs/2408.03825

### Indirectly Related Papers
- **“KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs” (Reiser et al., ICCV 2021)**:
  - Splits NeRF into small MLPs, suitable for NPU.
  - **NPU Relevance**: Inspires MLP-based 3DGS approaches.
  - **Source**: https://arxiv.org/abs/2103.13444
- **“Fast Dynamic Radiance Fields with Time-Aware Neural Voxels” (Fang et al., SIGGRAPH Asia 2022)**:
  - Lightweight radiance field model for low-power hardware.
  - **NPU Relevance**: Suggests NPU feasibility for 3D tasks.
  - **Source**: https://arxiv.org/abs/2205.15285

### Research Gaps
- No direct NPU-based 3DGS implementations.
- Challenges: NPU’s limited support for graphics tasks (sorting, rasterization).
- Opportunities: Leverage MLP-based approaches (e.g., NeuralGS) and compression (e.g., LightGaussian) for NPU compatibility.

---

## 3. Recommendations for NPU Implementation
1. **Select NPU**: Choose hardware (e.g., Apple M1/M2, Google TPU) and framework (PyTorch MPS, TensorFlow TPU).
2. **Port Existing Code**: Adapt gsplat or original 3DGS code for NPU.
3. **Hybrid Approach**: Use NPU for matrix operations/optimization, CPU for sorting/rasterization.
4. **Test Small Scenes**: Start with low-resolution scenes to manage NPU memory limits.
5. **Optimize**: Reduce Gaussian count, use compression techniques (e.g., LightGaussian).

---

## 4. Conclusion
Implementing 3DGS on NPUs is feasible but requires adapting CUDA-based workflows to NPU’s strengths (matrix operations, optimization) and offloading graphics tasks to CPU or custom ops. Research like NeuralGS and LightGaussian suggests NPU compatibility through MLP and compression. Start with gsplat and test on small scenes for best results.

For further details or specific NPU implementation (e.g., Apple Neural Engine), refer to the original conversation or contact the author.

---
*Generated by Grok 3, xAI, on July 16, 2025, 04:53 PM KST.*