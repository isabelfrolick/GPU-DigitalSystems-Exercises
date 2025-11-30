# CUDA Machine Problems

This repository contains my solutions to four CUDA-based machine problems completed on GPU servers (each equipped with NVIDIA Tesla C2075 GPUs). All work was developed and tested using Visual Studio 2015 and the CUDA toolkit, following the provided GPU CUDA Environment guidelines.

The projects explore device querying, parallel matrix addition, and dense matrix multiplication, with a focus on GPU architecture, execution configuration, and performance analysis.

---

## Machine Problem 1 — **CUDA Device Query**
Implemented a CUDA program to query and report the capabilities of the GPU servers.  
The program outputs key hardware characteristics, including:

- Number and type of CUDA devices  
- Clock rate  
- Number of streaming multiprocessors (SMs)  
- Number of CUDA cores  
- Warp size  
- Global memory size  
- Constant memory size  
- Shared memory per block  
- Registers per block  
- Maximum threads per block  
- Max block dimensions  
- Max grid dimensions  

Ensured that the CUDA toolchain was functioning correctly and introduced me to GPU architecture parameters.

---

## Machine Problem 2 — **Matrix Addition (GPU vs CPU)**
Implemented several versions of a matrix addition kernel for square integer matrices, and comparing GPU performance with a CPU reference implementation.

### Implemented Kernels
1. **One thread per output element**  
2. **One thread per output row** (16 threads per block)  
3. **One thread per output column** (16 threads per block)

### Work Completed
- Wrote host code for memory allocation, data transfer, kernel launch, output retrieval, and verification.  
- Implemented all three kernel variants using appropriate execution configurations (including 16×16 thread blocks).  
- Compared device results against CPU output and validated correctness (“Test PASSED”).  
- Benchmarked GPU kernels and CPU execution for matrix sizes:  
  - 16×16  
  - 256×256  
  - 4096×4096  
- Compiled tables/graphs comparing GPU vs CPU runtime.  
- Analyzed the performance trade-offs and advantages/disadvantages of each kernel design.

---

## Machine Problem 3 — **Matrix Multiplication & Data Transfer Analysis**
Extended the previous work to dense matrix multiplication, emphasizing both computation and data transfer overhead.

### Work Completed
- Implemented a GPU matrix multiplication kernel (one thread per output element).  
- Wrote host-side memory allocation, transfer, kernel launch, comparison with CPU results, and cleanup.  
- Measured **host→device** and **device→host** transfer time for matrix sizes:  
  - 16×16  
  - 256×256  
  - 4096×4096  
- Benchmarked GPU and CPU multiplication times using various block widths:  
  - 1  
  - 4  
  - 16  
  - 32  
- Produced plots for:
  - Data transfer time vs. matrix size  
  - GPU vs CPU compute time  
  - Kernel execution time vs. block-width / number of blocks  
  - Total application time (including transfers)

### Analysis Highlights
- Evaluated when GPU offloading is beneficial once data transfer cost is included.  
- Discussed how varying block sizes and thread-per-block configurations affect performance and resource utilization.  
- Addressed questions such as:  
  - How often each matrix element is loaded  
  - The impact of memory access patterns  
  - Why certain block sizes yield better performance on the Tesla C2075 architecture  

## Machine Problem 4 — Tiled Matrix Multiplication (CUDA)

My implementation of a shared-memory–optimized, tiled matrix multiplication kernel. The work builds on earlier assignments by introducing shared memory tiling to improve data locality and GPU performance.

The kernel computes the product:

\[
P = M \times N
\]

for square integer matrices of size 16×16, 256×256, and 4096×4096.

Each GPU thread computes one output element, and shared memory tiles are used to reduce redundant global memory accesses.

---
