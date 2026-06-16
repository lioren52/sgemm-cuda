# ⚡ CUDA GEMM Optimization Engine

A bare-metal C++ CUDA project dedicated to pushing General Matrix Multiplication (GEMM) to the absolute hardware limit. 

Because why let PyTorch have all the fun when you can write the backend yourself? 

This repository serves as a testing ground for low-level GPU architecture optimizations, memory coalescing, and thread-level parallelism. The goal is to incrementally transform a naive matrix multiplication algorithm into a highly tuned engine capable of saturating the SMs of an NVIDIA GPU.

## 📊 Performance Tracker

This project is built on the philosophy of progressive optimization. Every architectural change to the kernel is benchmarked and verified for mathematical accuracy against a standard CPU implementation. 

**Test Dimensions:** `1024 x 512` multiplied by `512 x 1024` (Output: 1,048,576 elements)
**Hardware Target:** NVIDIA Tesla T4 

| Kernel Version | Algorithm Strategy | Time (ms) | Throughput (GFLOPs) | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0** | Naive Global Memory | `2.909` | `369.05` | ✅ Verified | Baseline implementation. Heavily memory-bound by standard VRAM latency. |
| **v2.0** | Shared Memory Tiling | `1.798` | `597.20` | ✅ Verified | 61% speedup. Bypassed VRAM latency by building L1 scratchpads. Currently bottlenecked by Shared Memory bandwidth (1:2 math-to-memory ratio). |
| **v3.0** | Thread Coarsening | *TBD* | *TBD* | 🚧 In Progress | Implementing Register Tiling to increase math intensity per thread, alter the memory ratio, and break the TeraFLOP barrier. |

## 🛠️ Build & Execute

This project uses standard NVIDIA compiler infrastructure without heavy abstractions. 

**Compile the engine:**
```bash
nvcc -arch=sm_75 v2-tiled.cu -o tiled
