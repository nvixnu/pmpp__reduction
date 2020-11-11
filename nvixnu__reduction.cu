#include "nvixnu__reduction.h"

__global__
void sum_by_block(double *v, double *sum, const int length){
    extern __shared__ double partial_sum[];
    unsigned int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;

    // Copies the elements to be added to the shared memory. If the value is a garbage one, includes zero instead
    partial_sum[tx] = tid < length ? v[tid] : 0.0;

    // Halve the stride in each iteration, bringing the temporary sums into the first half
    for(unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if(tx < stride){ // Check if thread is inthe first half
            partial_sum[tx] += partial_sum[tx+stride];
        }
    }
    __syncthreads();
    sum[blockIdx.x] = partial_sum[0];    
}