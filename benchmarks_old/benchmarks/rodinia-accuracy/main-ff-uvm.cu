#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include <cub/cub.cuh>
#include "reference.h"

#include "../../FlexFloat.h"

#define GPU_NUM_THREADS 256

__global__
void accuracy_kernel(
    const int N,
    const int D,
    const int top_k,
    FFArr32 Xdata,
    const int* labelData,
    int* accuracy)
{
  typedef cub::BlockReduce<int, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int label = labelData[row];
    const float label_pred = Xdata[row * D + label];
    int ngt = 0;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      const float pred = Xdata[row * D + col];
      if (pred > label_pred || (pred == label_pred && col <= label)) {
        ++ngt;
      }
    }
    ngt = BlockReduce(temp_storage).Sum(ngt);
    if (ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { 
    atomicAdd(accuracy, count);
  }
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
  free_bytes -= 4096 * 4;

  const double factor = 3.0;
  const size_t array_elements = free_bytes * factor / 4;
  const size_t array_size_in_bytes = array_elements * 4;

  const long long native_difference = free_bytes - array_size_in_bytes;
  printf("GPU memory: %zu free out of %zu total\n", free_bytes, total_bytes);
  printf("Native memory footprint: %zu bytes :: %lld byte difference from available memory\n", array_size_in_bytes, native_difference);

  size_t user_ff_bits_array[] = {32, 31, 28, 24, 20, 17};

  for (int i = 0; i < sizeof(user_ff_bits_array) / 8; i++) {
    const size_t user_ff_bits = user_ff_bits_array[i];
    const size_t ff_array_elements = get_ffarray_size_float(array_elements, user_ff_bits);
    const size_t ff_array_size_in_bytes = ff_array_elements * 4;

    const long long fitfloat_difference = array_size_in_bytes - ff_array_size_in_bytes;
    const long long fitfloat_difference_available = free_bytes - ff_array_size_in_bytes;
    printf("FF-%zu  memory footprint: %zu bytes :: %lld byte difference from native arrays, %lld from available memory\n", user_ff_bits, ff_array_size_in_bytes, fitfloat_difference, fitfloat_difference_available);    
  }

  const size_t nrows = array_elements / 4096;
  const size_t ndims = 4096;
  const size_t top_k = 64;
  const size_t repeat = 9;

  const size_t data_size = nrows * ndims;

  const size_t label_size_bytes = nrows * sizeof(int); 
  const size_t data_size_bytes = data_size * sizeof(float);

  printf("Using %.2f GB for data and %f GB for labels\n", 1.0 * data_size_bytes / 1000000000, 1.0 * label_size_bytes / 1000000000);

  int *label = (int*) malloc (label_size_bytes);

  srand(123);
  for (int i = 0; i < nrows; i++)
    label[i] = rand() % ndims; 

  float *data = (float*) malloc (data_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (0.f, 1.f);
  for (int i = 0; i < data_size; i++) {
    data[i] = distr(g);
  }

  int *d_label;
  cudaMalloc((void**)&d_label, label_size_bytes);

  FFArr32 d_data(data_size);

  int *d_count;
  cudaMalloc((void**)&d_count, sizeof(int));

  cudaMemcpy(d_label, label, label_size_bytes, cudaMemcpyHostToDevice);
  FloatH2FFDmemcpy(d_data, data, data_size);
  cudaDeviceSynchronize();
  dim3 block (GPU_NUM_THREADS);

  double rts[repeat];
  for (int ngrid = nrows; ngrid <= nrows; ngrid += nrows / 4) {

    dim3 grid (ngrid);

    for (int i = 0; i < repeat; i++) {
      auto start=std::chrono::steady_clock::now();
      cudaMemset(d_count, 0, sizeof(int));
      accuracy_kernel<<<grid, block>>>(nrows, ndims, top_k, d_data, d_label, d_count);
      cudaDeviceSynchronize();
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
      rts[i] = time * 1e-9f;
    }
    
    printf("~ accuracy: %f (s)\n", rts[repeat / 2]);

    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Accuracy = %f\n", (float)count / nrows);
  }

  cudaFree(d_label);
  cudaFree(d_count);

  free(label);
  free(data);

  return 0;
}
