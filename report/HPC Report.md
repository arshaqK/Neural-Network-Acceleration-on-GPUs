# HPC Project: Neural Network Acceleration on GPUs

### Team Members: Muaz Ahmed (22i-1125) | Arshaq Kirmani (i220834) | Ibtehaj Haider (22i-0767)

### GitHub Repository: https://github.com/arshaqK/Neural-Network-Acceleration-on-GPUs/tree/master (Main for Project Submission)

---

## Introduction

This report explores the performance acceleration of a simple fully connected neural network on the MNIST dataset using GPU-based parallelization. The goal was to start from a sequential CPU baseline and improve its performance across multiple GPU-accelerated versions, ultimately leveraging CUDA and tensor cores.

---

## Dataset & Model Overview

- **Dataset:** MNIST (28x28 grayscale images, 10 classes, 70k total samples)
    
- **Model:**
    
    - Input size: 784
        
    - Hidden layer: 128 neurons (ReLU)
        
    - Output layer: 10 neurons (Softmax)
        
    - Training: Categorical Cross-Entropy + SGD
        

---

## Version Overview

### **V1: Sequential CPU Implementation**

- **Description:** Pure C-based implementation without any parallelization.
    
- **Compiler:** GCC with `-O2` and `-pg` flags for profiling.
    
- **Profiling Tool:** gprof
    
- **Execution Time:** ~[Insert total seconds here]
    
- **Key Bottlenecks (gprof):**
    
    - `forward()`: [44.363%]
        
    - `backward()`: [49.039%]
        

**Observations:**  
CPU-bound operations, particularly matrix multiplications and weight updates, dominate time complexity. No use of SIMD or threading.

---

### **V2: Naive GPU Implementation (CUDA)**

- **Strategy:** Ported forward and backward passes to CUDA kernels with 1D thread blocks.
    
- **Execution Time:** ~[Insert time]
    
- **Speedup over V1:** [x.x]x
    
- **Issues:**
    
    - Inefficient global memory access
        
    - Poor thread occupancy
        
    - No shared memory usage
        

**Observations:**  
Although we saw initial speedups, naive memory access patterns and launch configurations severely limited performance. No significant improvement in floating-point throughput.

---

### **V3: Optimized GPU Implementation**

- **Optimizations Added:**
    
    - Better launch configurations (block/thread tuning)
        
    - Shared memory for intermediate values
        
    - Memory coalescing
        
    - Reduced memory transfers between host and device
        
- **Execution Time:** ~[Insert time]
    
- **Speedup over V1:** [x.x]x
    
- **Speedup over V2:** [x.x]x
    

**Profiling (Nsight Compute or nvprof):**

- Global memory throughput: [Insert MB/s]
    
- Occupancy: [Insert %]
    
- Instruction throughput: [Insert % of peak]
    

**Observations:**  
Substantial improvements in execution speed. Most major bottlenecks related to memory hierarchy were addressed. Kernel execution became more efficient and uniform.

---

### **V4: Tensor Core Optimization**

- **Changes:**
    
    - Used Tensor Cores via WMMA API for matrix multiplications
        
    - Converted float32 operations to half-precision where applicable
        
    - Alignment of dimensions to 16x16 tile requirements
        
- **Execution Time:** ~[Insert time]
    
- **Speedup over V1:** [x.x]x
    
- **Speedup over V3:** [x.x]x
    

**Limitations:**

- Tensor cores require strict layout adherence
    
- Half-precision impacted accuracy slightly, but remained within tolerable bounds
    

**Observations:**  
This version offered the best performance, especially during forward pass due to use of FP16 tensor core acceleration. However, the improvements were limited by input/output conversion overhead and precision trade-offs.

---

## Summary of Results (Example Table)

| Version | Execution Time (s) | Speedup vs V1 | Accuracy (%) |
| ------- | ------------------ | ------------- | ------------ |
| V1      | [ ]                | 1x            | [ ]          |
| V2      | [ ]                | [ ]           | [ ]          |
| V3      | [ ]                | [ ]           | [ ]          |
| V4      | [ ]                | [ ]           | [ ]          |

---

## Conclusion

This project demonstrated how simple neural networks can benefit immensely from GPU acceleration. Starting from a sequential baseline, each version introduced new layers of optimization:

- V2 introduced parallelism,
    
- V3 tackled memory and kernel launch issues,
    
- V4 leveraged specialized hardware for maximum speed.
    

**Remaining Bottlenecks:**

- Weight updates in backprop still have room for kernel-level parallelization.
    
- Accuracy vs. precision trade-off in FP16.
    

---

## 🔗 Recommendations for Future Work

- Convert to a mini-batch version with shared memory for matrix accumulation
    
- Add support for multi-GPU training with NCCL or OpenMPI
    
- Explore cuDNN or PyTorch-C++ for high-level abstraction
