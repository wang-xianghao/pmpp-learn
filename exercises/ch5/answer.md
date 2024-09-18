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

There are $1000 \times $