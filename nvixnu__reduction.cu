#include "nvixnu__reduction.h"

__global__
void nvixnu__sum_by_block(float *v, float *sum){
    extern __shared__ float partial_sum[];
    unsigned int tx = threadIdx.x;

    partial_sum[threadIdx.x] = v[blockIdx.x * blockDim.x + tx];

    for(unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if(tx < stride){
            partial_sum[tx] += partial_sum[tx+stride];
        }
    }
    __syncthreads();
    sum[blockIdx.x] = partial_sum[0];
}