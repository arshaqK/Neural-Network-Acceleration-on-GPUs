# 🧠 Neural Network Acceleration on GPUs (HPC Final Project)

**Course:** High Performance Computing  
**Team Members:** Muaz Ahmed · Arshaq Kirmani · Ibtehaj Haider  
**Repository:** [GitHub Link](https://github.com/arshaqK/Neural-Network-Acceleration-on-GPUs/tree/master)  
**Report:** [Report Link](https://drive.google.com/file/d/1PDlFQCK2RdJ0WdXa9qTWxkOnqOxZkh4v/view?usp=sharing)<br>
**Presentation:** [Google Slides](https://docs.google.com/presentation/d/1MFKEoj1a91RdgRyF7rPHOS-PPBuwbGjqRf7nvhiuoKc/)

---

## 🚀 Overview

This project aimed to accelerate the training of a simple feedforward neural network on the MNIST dataset using GPU parallelism. We incrementally optimized our implementation from a basic CPU version to more advanced GPU-accelerated versions using:

- **CUDA (baseline + optimized kernels)**
- **Tensor Cores (via WMMA API)**
- **OpenACC directives for easier portability and development**

The ultimate goal was to investigate how performance scales when introducing different layers of GPU acceleration — and what trade-offs exist between speed and model accuracy.

---

## 🗂 Dataset & Model

- **Dataset:** MNIST (70,000 grayscale images, 10 digits, 28x28 pixels)
- **Model Architecture:**
  - Input Layer: 784 neurons (flattened image)
  - Hidden Layer: 128 neurons with ReLU activation
  - Output Layer: 10 neurons with Softmax
  - Loss Function: Categorical Cross-Entropy
  - Optimizer: Stochastic Gradient Descent

---

## 🔧 Version Breakdown

### 🔹 V1: Sequential CPU (Baseline)
- Written in C with no parallelism.
- Profiled using `gprof`.
- Execution Time: **97.5s**
- Accuracy: **97.9%**

### 🔹 V2: Naive CUDA Implementation
- Ported forward and backward passes to CUDA with basic kernel launches.
- Improved memory access patterns, vectorization, and batch-level computation.
- Execution Time: **22.494s**
- Speedup: **4.33x**
- Accuracy: **97.9%**

### 🔹 V3: Optimized CUDA
- Used shared memory, tiled matrix operations, and reduced memory transfers.
- Significant speedup with minor loss in accuracy (due to batch size optimization).
- Execution Time: **7.2s**
- Speedup: **13.54x**
- Accuracy: **87.21%**

### 🔹 V4: Tensor Core (WMMA API)
- Intended to leverage Tensor Cores using half-precision (FP16).
- Alignment to 16x16 matrix tiles was implemented.
- Testing not completed due to system constraints.
- **Execution Time / Accuracy:** *N/A*

### 🔹 V5: OpenACC (Bonus Version)
- Used `#pragma acc` for directive-based parallelism.
- Focused on developer productivity and code maintainability.
- Execution Time: **19.98s**
- Speedup: **4.87x**
- Accuracy: **92.53%**

---

## 📊 Results Summary

| Version    | Execution Time (s) | Speedup vs V1 | Accuracy (%) |
| ---------- | ------------------ | ------------- | ------------ |
| V1         | 97.5               | 1x            | 97.9         |
| V2         | 22.494             | 4.33x         | 97.9         |
| V3         | 7.2                | 13.54x        | 87.21        |
| V4         | N/A                | N/A           | N/A          |
| V5 (Bonus) | 19.98              | 4.87x         | 92.53        |

All performance benchmarks were conducted on **university HPC servers**, ensuring accurate and reproducible results.

---

## ⚠️ Challenges Faced

- Navigating CUDA’s low-level memory management and synchronization semantics.
- Aligning matrix dimensions for Tensor Core compatibility.
- Accuracy drops from FP32 → FP16 conversions in Tensor Core version.
- Tuning kernel configurations (block sizes, shared memory, register pressure).
- Limited access to long runtime windows on university GPUs.

---

## 📚 Key Learnings

This project was a crash course in practical parallel programming for machine learning. It sharpened our understanding of:

- GPU memory hierarchies (global, shared, constant)
- CUDA kernel optimization techniques
- The trade-off between precision and speed (especially with FP16)
- Ease of use and portability with OpenACC
- Profiling & debugging tools for performance tuning (e.g., `nvprof`, `gprof`)

---

## 🔮 Future Improvements

- Add true mini-batch support with better shared memory accumulation.
- Implement multi-GPU training using NCCL or MPI.
- Integrate cuDNN or switch to PyTorch-C++ frontend for abstraction.
- Explore scheduling techniques and learning rate decay for improved convergence.

---

## 🧠 Final Thoughts

This project pushed us out of our comfort zones — from basic C to hardcore CUDA kernels to tensor core madness. Each version made us appreciate how much performance is *actually* sitting under the hood, waiting to be unlocked by good parallel design.

We learned a lot. We cursed a lot. But most importantly, we finished strong.

---

Made with coffee, CUDA, and late-night debugging ✨

---
