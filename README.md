# ⚡ CUDA SGEMM Optimization Engine

A bare-metal C++ CUDA project dedicated to pushing Single-Precision General Matrix Multiplication (SGEMM) to the absolute hardware limit. 

Because why let PyTorch have all the fun when you can write the backend yourself? 

This repository serves as a systematic teardown of low-level GPU architecture. It incrementally transforms a naive matrix multiplication algorithm into a highly tuned engine implementing warp-level cooperative fetching, 2D register tiling, and 128-bit vectorized loads. The final custom kernel achieves **>52% of NVIDIA's proprietary cuBLAS performance** on virtualized hardware without using a single line of assembly or Tensor Cores.

## 📊 Benchmarking & Environment

This project is built on the philosophy of progressive optimization. Every architectural change is benchmarked, isolated, and verified for mathematical accuracy against a standard CPU implementation. 

* **Test Dimensions:** `4096 x 4096` multiplied by `4096 x 4096` (Output: 16,777,216 elements)
* **Hardware Target:** NVIDIA Tesla T4 (Virtualized)
* **Environment:** Tested on free-tier Google Colab and Kaggle instances.
    * *Note on Cloud Throttling:* Virtualized environments heavily throttle power draw and clock speeds. NVIDIA's own cuBLAS library peaks at ~4.2 TFLOPs on this environment (out of the T4's 8.1 TFLOPs theoretical peak). Therefore, success in this repository is measured **relative to cuBLAS**, not the theoretical silicon limit.

## 🚀 The Optimization Progression

| Kernel Version | Algorithm Strategy | Median Time (ms) | Throughput (GFLOPs) | Relative to cuBLAS | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0** | Naive Global Memory | `322.34` | `426.37` | `10.1%` | ✅ Verified |
| **v2.0** | Shared Memory Tiling | `216.29` | `635.42` | `15.0%` | ✅ Verified |
| **v3.0** | 1D Thread Coarsening | `118.43` | `1160.47` | `27.4%` | ✅ Verified |
| **v4.0** | 2D Thread Coarsening | `88.78` | `1547.95` | `36.6%` | ✅ Verified |
| **v5.0** | Warp Tiled (1D Linearized) | `71.79` | `1914.22` | `45.3%` | ✅ Verified |
| **v6.0** | Vectorized Loads (`float4`) | `62.37` | `2203.46` | `52.2%` | ✅ Verified |
| **Ref** | **NVIDIA cuBLAS** | `32.56` | `4219.91` | `100.0%` | 🟢 Baseline |

## 🏗️ Architectural Teardown

Here is exactly how the silicon bottlenecks were identified and eliminated at each stage:

### v1.0: Naive Global Memory
The baseline implementation. Functionally correct and naturally memory-coalesced, but heavily bottlenecked by standard VRAM latency. Every math operation requires a slow trip to Global Memory.

### v2.0: Shared Memory Tiling
Introduced the L1 Shared Memory scratchpad. Blocks of threads collaboratively load tiles of Matrix A and B into ultra-fast SRAM before computing. This drastically reduces VRAM round-trips, yielding a ~50% speedup over the naive approach.

### v3.0 & v4.0: 1D & 2D Thread Coarsening (Register Tiling)
Shifted the bottleneck from the L1 cache to the ALU math units. By allocating physical silicon registers (`pVal`, `regA`, `regB`) and looping the dot-product calculations on the *outside* of the fetch loop (Outer Product), the arithmetic intensity (compute-to-memory ratio) skyrocketed. However, scaling to larger block sizes choked the SM occupancy due to massive register pressure per thread.

### v5.0: Warp Tiling & 1D Linearization
Completely decoupled the memory loading phase from the math execution phase. Threads are flattened into a 1D marching line to act as a cooperative bucket brigade, loading data into Shared Memory without uncoalesced gaps. Once loaded, Warps cooperatively broadcast data into private registers. This allowed the thread block size to shrink to `16x16`, dropping register pressure, spiking SM occupancy, and pushing the engine to 1.9 TFLOPs.

### v6.0: 128-bit Vectorized Loads
The final memory pipeline optimization. Recast memory pointers to `float4`, commanding the SM to fetch 128-bit chunks per thread instead of standard 32-bit floats. This replaced four separate `LDG.E` assembly instructions with a single `LDG.E.128` instruction, slashing the instruction overhead for memory fetches by 75%, unchoking the warp scheduler, and breaking the 50% cuBLAS barrier.

## 🛠️ Build & Execute

This repository isolates the host benchmarking driver from the device kernels for clean compilation. To compile the entire benchmarking suite, ensure you have the CUDA Toolkit installed and link the cuBLAS library.

**Compile:**
```bash
nvcc benchmark.cu \
     src/utils.cpp \
     src/kernels/*.cu \
     -Iinclude \
     -lcublas \
     -O3 \
     -arch=sm_75 \
     --use_fast_math \
     -lineinfo \
     -o sgemm_benchmark
