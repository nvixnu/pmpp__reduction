
#ifndef NVIXNU__REDUCTION_H_
#define NVIXNU__REDUCTION_H_

/**
* Kernel that sums elements of an array
* @param v The input array
* @param sum The array with the partial sums values (Sums of each block).
*/
__global__ void nvixnu__sum_by_block(float *v, float *sum);

#endif /* NVIXNU__REDUCTION_H_ */