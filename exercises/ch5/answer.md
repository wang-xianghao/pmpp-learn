# Chapter 5

## Question 1 
**Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.**

No, matrix addition is element-wise, there is no common accessed elements between threads.

## Question 2
**Draw the equivalent of Fig. 5.7 for a $8\times 8$ matrix multiplication with $2 \times 2$ tiling and $4 \times 4$ tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.**

For naive matrix multiplication, total $8\times 8 \times (2\times 8)=1024$ data is loaded from the global memory. In the $2\times 2$ tiling optimization, one tile loads $2 \times 2 \times 8 = 32$ data. With $\frac{8\times 8}{2\times 2}=16$ tiles, $16 \times 32 = 512$ data is required to be loaded from the global memory. Compared to the naive one, $2 \times 2$ tiling requires $0.5\times$ global memory reads.

Similarly, one $4 \times 4$ tile loads $2 \times 4 \times 8 = 64$ data. With $\frac{8\times 8}{4 \ times 4} = 4$ tiles, $4 \times 64 = 256$ data is loaded from the global memory. This requires $0.25 \times$ global memory loads.


<p align="center">
<img src="./pmpp_tiling.png" alt="drawing" width="500"/>
</p>

## Question 3
**What type of incorrect execution behavior can happen if one forgot to use one or both __syncthreads() in the kernel of Fig. 5.9?**

*Fig. 5.9*
```C
#define TILE_WIDTH 16

void matrixMulKernel (float* M, float* N, float* P, int Width) {
    __shared__ float Mds [TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds [TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        // Collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty) * Width + Col];
        __syncthreads() ;
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads() ;
    }
    P[Row*Width + Col] = Pvalue;
}
```

1. The synchronization ensures threads within a block load all the required elements into the shared memory for computation in the current phase. If missing this, for example, $P_{0,0}$ starts the computation without waiting for $P_{1,0}$. The element $N_{1,0}$ required by $P_{0,0}$'s calculation is not loaded into the shared memory. Thus, the calculation result might be wrong.

2. This synchronization prevents the required elements in shared memory is overwritten by next phase. For example, if $P_{0,0}$ is calculating $M_{0,0}\times N{0,0} + M_{0,1} \times N_{1,0}$. With the synchronization, $P_{1,0}$, which already finished the phase 0, replaces $M_{1,0}$ by $M_{1,2}$.

## Question 4
**Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.**

Shared memory is shared across threads within a block. We can use shared memory to reduce some common memory accesses by these threads. By contrast, registers are private to each thread.

## Question 5
**For our tiled matrix-matrix multiplication kernel, if we use a $32 \times 32$ tile, what is the reduction of memory bandwidth usage for input matrices M and N?**

As discussed in Question 2, the memory bandwidth usaged reduces to $0.03125 \times$.

## Question 6
**Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?**

There are $1000 \times 512$ versions of this variable being created, because the scope of local variable is thread.

## Question 7
**In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?**

There are $1000$ versions of this varible being created, because the scope of shared memory variable is block.

## Question 8
**Consider performing a matrix multiplication of two input matrices with dimensions $N \times N$. How many times is each element in the input matrices requested from global memory when:**

**a. There is no tiling?** Let the computation be $C=A\times B$. The element $A_{i,j}$ is involved in computing the $i$-th row of $C$, so it is loaded $N$ times. Similar to $B$, each element of $B$ is loaded $N$ times by computing one column of $C$. Thus, each element in input matrices is requsted from global memory $N$ times.

**b. Tiles of size $T\times T$ are used?** We only consider $A$. The element $A_{i,j}$ is for computing the $i$-th row of $C$. With tiled $C$, $A_{i,j}$ is only loaded once within a tile, thus $A_{i,j}$ is loaded $\frac{N}{T}$ times.

## Questino 9
**A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory-bound.**

The arithmetic intensity is $\frac{36\ FLOPs/s}{7\times 4\ Bytes}=\frac{9}{7}\ FLOPs/Bytes$.

**a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second**
$\frac{9}{7}\times 100\times 10^{9}\approx128.6 \times 10^9 < 200 \times 10^9$. Thus, it is memory bound.

**b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second**
$\frac{9}{7}\times 250\times 10^9 \approx 321.4\times 10^9 > 300\times 10^9$. Thus, it is compute-bound.

## Question 10
**To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size `BLOCK_WIDTH` by `BLOCK_WIDTH`, and each of the dimensions of matrix A is known to be a multiple of `BLOCK_WIDTH`. The kernel invocation and code are shown below. `BLOCK_WIDTH` is known at compile time and could be set anywhere from 1 to 20.**

```c
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>> (A, A_width, A_height);

__global__
void BlockTranspose (float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```

**a. Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on the device?** 

`BLOCK_SIZE` should be at most $32$. The code misses synchronization after loading element to the shared memory. The last write may write trash value from `blockA` to the `A_elements`.

**b. If the code does not execute correctly for all `BLOCK_SIZE` values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all `BLOCK_SIZE` values.**

The root cause is mentioned above. We can simply add a block synchronization after writing to `blockA`. Another optimization trick is that we do not need shared memory, a variable in register is enough.
```C
__global__
void BlockTranspose (float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int baseIdx = row * A_width + col;
    int transIdx = col * A_width + row; 

    int transVal = A_elements[transIdx];
    __syncthreads();
    A_elements[baseIdx] = transVal;
}
```

## Question 11
Consider the following CUDA kernel and the corresponding host function that calls it:
```C
__global__ void foo_kernel(float* a, float* b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];
    for (unsigned int j = 0; j < 4; ++j) {
        x[j] = a[j * blockDim.x * gridDim.x + 1];
    }
    if (threadIdx.x == 0) {
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5f * x[0] + 3.7f * x[1] + 6.3f * x[2] + 8.5f * x[3]
        + y_s * b_s[threadIdx.x] + b_s[(threadIdx.x + 3) % 128];
}

void foo (int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<(N + 128 - 1) / 128, 128>>> (a_d, b_d);
}
```

**a. How many versions of the variable i are there?** $1024$

**b. How many versions of the array x[] are there?** $1024$

**c. How many versions of the variable y_s are there?** $\frac{1024}{128}=8$ 

**d. How many versions of the array b_s[] are there?** $\frac{1024}{128}=8$ 

**e. What is the amount of shared memory used per block (in bytes)?** `y_s` is $4$ bytes and $b_s$ is $128 \times 4=512$ bytes. Thus, each bock uses $4+512=518$ bytes of shared memory.

**f. What is the floating-point to global memory access ratio of the kernel (in OP/B)?**

There are $6$ global memory accesses ($1$ write, $5$ reads), and $10$ floating point opeartions. Thus, the ration is $\frac{10}{6\times 4}\approx0.42$ OP/B.

**12. Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.**

**a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.**
We need $\frac{2048}{64}=32$ blocks. 
The limiting factor is shared memory because $32 \times 4=128 > 96$.

**b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM.**
It can acheive the full occupancy. We need $\frac{2048}{256}=8$ blocks. The register is not a limit: $8 \times 256\times 31=63488 \le 65536$. The shared memory is not either a limit: $8\times 8=64 \le 96$.