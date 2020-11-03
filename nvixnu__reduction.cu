#include "nvixnu__reduction.h"

__global__
void nvixnu__sum_by_block(float *v, float *sum){
    extern __shared__ float partial_sum[];
    unsigned int tx = threadIdx.x;
    // Copies the elements to be added to the shared memory
    partial_sum[tx] = v[blockIdx.x * blockDim.x + tx];

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