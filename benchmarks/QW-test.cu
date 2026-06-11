
#include <algorithm>
#include <cassert>

#include "../FitFloat.h"


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


__global__ void vector_add_1(float* a, float* b, float* c, const size_t num_elements)
{
  const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < num_elements) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void vector_add_2(FFArr32 a, FFArr32 b, FFArr32 c, const size_t num_elements)
{
  const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < num_elements) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void vector_add_3(FFArr32 a, FFArr32 b, FFArr32 c, const size_t num_elements)
{
  const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < num_elements) {
    c.quick_write(idx, a[idx] + b[idx]);
  }
}

int main()
{
  // query memory

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);

  size_t reserved_size = free_bytes - ((size_t) 1024) * 1024 * 1024;

  int* reserved = nullptr;
  cudaMalloc((void**) &reserved, reserved_size);
  cudaMemset(reserved, 1, reserved_size);
  cudaDeviceSynchronize();

  cudaMemGetInfo(&free_bytes, &total_bytes);

  const size_t num_elements = free_bytes * 1.1 / 4 / 3; // 3 arrays

  const int num_runs = 9;
  const int thread_blocks = (num_elements + FF_TPB - 1) / FF_TPB;

  GPUTimer d_timer;
  double* runtimes = new double[num_runs];

  // native

  float* a1;
  float* b1;
  float* c1;

  cudaMallocManaged(&a1, num_elements * sizeof(float));
  cudaMallocManaged(&b1, num_elements * sizeof(float));
  cudaMallocManaged(&c1, num_elements * sizeof(float));

  for (int r = 0; r < num_runs; r++) {
    d_timer.start();
    vector_add_1<<<thread_blocks, FF_TPB>>>(a1, b1, c1, num_elements);
    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();
  }
  std::sort(runtimes, runtimes + num_runs);
  const double native_runtime = runtimes[num_runs / 2];
  printf("~ native: %6f s\n", native_runtime);

  cudaFree(a1);
  cudaFree(b1);
  cudaFree(c1);

  // FF

  FFArr32 a2(num_elements);
  FFArr32 b2(num_elements);
  FFArr32 c2(num_elements);

  for (int r = 0; r < num_runs; r++) {
    d_timer.start();
    vector_add_2<<<thread_blocks, FF_TPB>>>(a2, b2, c2, num_elements);
    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();
  }
  std::sort(runtimes, runtimes + num_runs);
  const double ff_runtime = runtimes[num_runs / 2];
  printf("~ ff    : %6f s, %.2fx\n", ff_runtime, native_runtime / ff_runtime);

  for (int r = 0; r < num_runs; r++) {
    d_timer.start();
    vector_add_3<<<thread_blocks, FF_TPB>>>(a2, b2, c2, num_elements);
    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();
  }
  std::sort(runtimes, runtimes + num_runs);
  const double ffqw_runtime = runtimes[num_runs / 2];
  printf("~ ff qw : %6f s, %.2fx, %.2fx\n", ffqw_runtime, native_runtime / ffqw_runtime, ff_runtime / ffqw_runtime);

  cudaFree(a1);
  cudaFree(b1);
  cudaFree(c1);

  return 0;
}