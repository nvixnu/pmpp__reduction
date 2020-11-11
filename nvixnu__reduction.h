
#ifndef NVIXNU__REDUCTION_H_
#define NVIXNU__REDUCTION_H_

/**
* Kernel that sums elements of an array
* @param v The input array
* @param sum The array with the partial sums values (Sums of each block).
* @param length The input array size
*/
__global__ void nvixnu__sum_by_block(double *v, double *sum, const int length);

#endif /* NVIXNU__REDUCTION_H_ */