#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda.h>
#include "../../FlexFloat.h"

#define threadsPerBlock  256

static inline __device__
void gpuAtomicAddNoReturn(int *address, int val) {
  atomicAdd(address, val);
}

#define FOR_KERNEL_LOOP(i, lim)                                      \
  for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < lim; \
                 i += gridDim.x * blockDim.x)

// Memory types used for the bincount implementations
enum class DeviceMemoryType { SHARED, GLOBAL };

template <typename input_t, typename IndexType>
__device__ static IndexType
getBin(input_t v, input_t minvalue, input_t maxvalue, IndexType nbins)
{
  IndexType bin = (v - minvalue) * nbins / (maxvalue - minvalue);
  // (only applicable for histc)
  // while each bin is inclusive at the lower end and exclusive at the higher,
  // i.e. [start, end) the last bin is inclusive at both, i.e. [start, end], in
  // order to include maxvalue if exists therefore when bin == nbins, adjust bin
  // to the last bin
  if (bin == nbins) bin--;
  return bin;
}

// Kernel for computing the histogram of the input
template <typename output_t,
          typename input_t,
          typename IndexType,
          DeviceMemoryType MemoryType>
__global__ void bincount (
       output_t *output,
  FFArr32 input,
  IndexType nbins,
  input_t minvalue,
  input_t maxvalue,
  IndexType input_size,
  IndexType output_size)
{
  extern __shared__ unsigned char my_smem[];
  output_t* smem = nullptr;

  if (MemoryType == DeviceMemoryType::SHARED) {
    // atomically add to block specific shared memory
    // then atomically add to the global output tensor
    smem = reinterpret_cast<output_t*>(my_smem);
    for (IndexType i = threadIdx.x; i < nbins; i += blockDim.x) {
      smem[i] = 0;
    }
    __syncthreads();

    FOR_KERNEL_LOOP(linearIndex, input_size) {
      const auto v = input[linearIndex];

      if (v >= minvalue && v <= maxvalue) {
        const IndexType bin = getBin<input_t, IndexType>(
                              v, minvalue, maxvalue, nbins);
        gpuAtomicAddNoReturn(&smem[bin], 1);
      }
    }
    __syncthreads();

    // Atomically update output bin count.
    for (IndexType i = threadIdx.x; i < nbins; i += blockDim.x) {
      gpuAtomicAddNoReturn(&output[i], smem[i]);
    }

  } else {
    ////////////////////////// Global memory //////////////////////////
    // atomically add to the output tensor
    // compute histogram for the block
    FOR_KERNEL_LOOP(linearIndex, input_size) {
      const auto v = input[linearIndex];
      if (v >= minvalue && v <= maxvalue) {
        const IndexType bin = getBin<input_t, IndexType>(
                              v, minvalue, maxvalue, nbins);
        gpuAtomicAddNoReturn(&output[bin], 1);
      }
    }
  }
}

#define HANDLE_CASE(MEMORY_TYPE, SHARED_MEM)                 \
  double rts[repeat]; \
  for (int i = 0; i < repeat; i++) {                         \
    auto start=std::chrono::steady_clock::now(); \
  bincount<                                                  \
      output_t,                                              \
      input_t,                                               \
      IndexType,                                             \
      MEMORY_TYPE><<<grid, block, SHARED_MEM>>>(             \
      d_output,                                              \
      d_input,                                               \
      nbins,                                                 \
      input_minvalue,                                        \
      input_maxvalue,                                        \
      input_size,                                            \
      output_size);                                          \
  cudaDeviceSynchronize();                                 \
  auto end = std::chrono::steady_clock::now();               \
  auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();\
  rts[i] = time * 1e-9f; } \
  printf("~ bincount: %f (s)\n", \
         (rts[repeat / 2]));

#define HANDLE_SWITCH_CASE(mType)                           \
  switch (mType) {                                          \
    case DeviceMemoryType::SHARED: {                        \
      HANDLE_CASE(DeviceMemoryType::SHARED, sharedMem);     \
      break;                                                \
    }                                                       \
    default: {                                              \
      HANDLE_CASE(DeviceMemoryType::GLOBAL, 0);             \
    }                                                       \
  }

/*
  Calculate the frequency of the input values.
  3 implementations based of input size and memory usage:
    case: sufficient shared mem
        SHARED: Each block atomically adds to it's own **shared** hist copy,
        then atomically updates the global tensor.
    case: insufficient shared memory
        GLOBAL: all threads atomically update to a single **global** hist copy.
 */
template <typename output_t, typename input_t, typename IndexType>
void eval(IndexType input_size, int repeat)
{
  size_t input_size_bytes = sizeof(input_t) * input_size;

  input_t *input = (input_t*) malloc (input_size_bytes); 

  // https://cplusplus.com/reference/random/normal_distribution/
  std::default_random_engine generator (123);
  std::normal_distribution<input_t> distribution(5.0,2.0);
  for (int i = 0; i < input_size; i++) {
    input[i] = distribution(generator);
  }

  auto min_iter = std::min_element(input, input+input_size);
  auto max_iter = std::max_element(input, input+input_size);
  
  input_t input_minvalue = *min_iter;
  input_t input_maxvalue = *max_iter;
  printf("Input min, max values: (%f %f)\n", (float)input_minvalue, (float)input_maxvalue);

  FFArr32 d_input(input_size);
  FloatH2FFDmemcpy(d_input, input, input_size);

  int maxSharedMemory;
  cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  printf("Maximum shared local memory size per block in bytes: %d\n", maxSharedMemory);

  for (IndexType nbins = 6144; nbins <= 768 * 32; nbins = nbins * 2) {

    printf("\nNumber of bins: %llu\n", nbins);
    IndexType sharedMem = nbins * sizeof(output_t);

    IndexType output_size = nbins;
    size_t output_size_bytes = sizeof(output_t) * output_size;
    output_t *output = (output_t*) malloc (output_size_bytes); 

    output_t *d_output;
    cudaMalloc((void**)&d_output, output_size_bytes);

    dim3 grid ((input_size + threadsPerBlock - 1) / threadsPerBlock);
    dim3 block (threadsPerBlock);

    // determine memory type to use in the kernel
    printf("bincount using global atomics\n");

    DeviceMemoryType memType = DeviceMemoryType::GLOBAL;
    cudaMemset(d_output, 0, output_size_bytes);
    HANDLE_SWITCH_CASE(memType)
    cudaMemcpy(output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);

    auto min_iter = std::min_element(output, output+output_size);
    auto max_iter = std::max_element(output, output+output_size);
    output_t minvalue = *min_iter;
    output_t maxvalue = *max_iter;
    printf("Output min, median, max values: (%ld %ld %ld)\n",
           (int64_t)minvalue / repeat,
           (int64_t)output[output_size/2] / repeat,
           (int64_t)maxvalue / repeat);

    if (sharedMem <= maxSharedMemory) {
      printf("\n");
      printf("bincount using global and local atomics\n");

      cudaMemset(d_output, 0, output_size_bytes);
      memType = DeviceMemoryType::SHARED;
      HANDLE_SWITCH_CASE(memType)
      cudaMemcpy(output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);

      auto min_iter = std::min_element(output, output+output_size);
      auto max_iter = std::max_element(output, output+output_size);
      output_t minvalue = *min_iter;
      output_t maxvalue = *max_iter;
      printf("Output min, median, max values: (%ld %ld %ld)\n\n",
             (int64_t)minvalue / repeat,
             (int64_t)output[output_size/2] / repeat,
             (int64_t)maxvalue / repeat);
    }

    cudaFree(d_output);
    free(output);
  }

  free(input);
}

int main(int argc, char* argv[])
{
  // query memory

  size_t reserved_size = ((size_t) 1024) * 1024 * 1024 * 22;

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  printf("GPU memory: %zu free out of %zu total\n", free_bytes, total_bytes);

  int* reserved = nullptr;
  cudaMalloc((void**) &reserved, reserved_size);
  cudaMemset(reserved, 1, reserved_size);
  cudaDeviceSynchronize();

  cudaMemGetInfo(&free_bytes, &total_bytes);

  const double factor = 2.0;
  const size_t array_elements = free_bytes * factor / 4 / 2; // 2 arrays
  const size_t array_size_in_bytes = array_elements * 4;
  
  const long long native_difference = free_bytes - array_size_in_bytes * 2;
  printf("GPU memory: %zu free out of %zu total\n", free_bytes, total_bytes);
  printf("Native memory footprint: %zu bytes :: %lld byte difference from available memory\n", array_size_in_bytes * 2, native_difference);

  size_t user_ff_bits_array[] = {32, 31, 28, 24, 20, 17};

  for (int i = 0; i < sizeof(user_ff_bits_array) / 8; i++) {
    const size_t user_ff_bits = user_ff_bits_array[i];
    const size_t ff_array_elements = get_ffarray_size_float(array_elements, user_ff_bits);
    const size_t ff_array_size_in_bytes = ff_array_elements * 4;

    const long long fitfloat_difference = array_size_in_bytes * 2 - ff_array_size_in_bytes * 2;
    const long long fitfloat_difference_available = free_bytes - ff_array_size_in_bytes * 2;
    printf("FF-%zu  memory footprint: %zu bytes :: %lld byte difference from native arrays, %lld from available memory\n", user_ff_bits, ff_array_size_in_bytes * 2, fitfloat_difference, fitfloat_difference_available);    
  }
  printf("Using %f GB\n", (1.0 * array_elements*sizeof(float)*2) / 1000000000);

  const unsigned long long n = array_elements;
  const int repeat = 9;

  eval<int, float, unsigned long long>(n, repeat);

  return 0; 
}
