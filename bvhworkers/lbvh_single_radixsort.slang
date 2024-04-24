#define WORKGROUP_SIZE 256// assert WORKGROUP_SIZE >= RADIX_SORT_BINS
#define RADIX_SORT_BINS 256
#define SUBGROUP_SIZE 32// 32 NVIDIA; 64 AMD

#define BITS 32// sorting uint32_t
#define ITERATIONS 4// 4 iterations, sorting 8 bits per iteration

[AutoPyBindCUDA]
[CUDAKernel]
void radix_sort(
    )
{
}