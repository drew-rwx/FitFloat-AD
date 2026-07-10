#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <cuda.h>
#include "reference.h"
#include "../../FitFloat.h"

#if defined(UVM1)
  #define factor 1.1
#else
  #define factor 3.0
#endif

template <typename T, typename G>
__global__
void adam (
        T* __restrict__ p,
        T* __restrict__ m,
        T* __restrict__ v,
  const G* __restrict__ g,
  const float b1,
  const float b2,
  const float eps,
  const float grad_scale,
  const float step_size,
  const int time_step,
  const size_t vector_size,
  adamMode_t mode,
  const float decay)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t totThreads = gridDim.x*blockDim.x;

  for (size_t j = i; j < vector_size; j += totThreads) {
    for (int t = 0; t < time_step; t++) {
      T scaled_grad = g[j]/grad_scale;
      m[j] = b1*m[j] + (1.f-b1)*scaled_grad;
      v[j] = b2*v[j] + (1.f-b2)*scaled_grad*scaled_grad;
      float m_corrected = m[j] / (1.f-powf(b1, t));
      float v_corrected = v[j] / (1.f-powf(b2, t));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sqrtf(v_corrected + eps);
      else // Mode 1
        denom = sqrtf(v_corrected) + eps;
      float update = (m_corrected/denom) + (decay*p[j]);
      p[j] -= (step_size*update);
    }
  }
}

int main(int argc, char* argv[])
{
  // query memory

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);

  size_t reserved_size = free_bytes - ((size_t) 1024) * 1024 * 1024 * 2;

  int* reserved = nullptr;
  cudaMalloc((void**) &reserved, reserved_size);
  cudaMemset(reserved, 1, reserved_size);
  cudaDeviceSynchronize();

  cudaMemGetInfo(&free_bytes, &total_bytes);

  const size_t array_elements = free_bytes * factor / 4 / 4; // four arrays
  const size_t array_size_in_bytes = array_elements * 4;

  const int time_step = 200;
  const int repeat = 3;

  float *m = (float*) malloc (array_size_in_bytes);
  float *v = (float*) malloc (array_size_in_bytes);
  float *g = (float*) malloc (array_size_in_bytes);
  float *p = (float*) malloc (array_size_in_bytes);

  srand(123);
  for (size_t i = 0; i < array_elements; i++) {
    m[i] = rand() / (float)RAND_MAX;
    v[i] = rand() / (float)RAND_MAX;
    g[i] = rand() / (float)RAND_MAX;
    p[i] = rand() / (float)RAND_MAX;
  }

  float *d_m, *d_v, *d_g, *d_p;

  cudaMallocManaged((void**) &d_m, array_size_in_bytes);
  cudaMallocManaged((void**) &d_v, array_size_in_bytes);
  cudaMallocManaged((void**) &d_g, array_size_in_bytes);
  cudaMallocManaged((void**) &d_p, array_size_in_bytes);

  cudaMemcpy(d_m, m, array_size_in_bytes, cudaMemcpyHostToHost);
  cudaMemcpy(d_v, v, array_size_in_bytes, cudaMemcpyHostToHost);
  cudaMemcpy(d_g, g, array_size_in_bytes, cudaMemcpyHostToHost);
  cudaMemcpy(d_p, p, array_size_in_bytes, cudaMemcpyHostToHost);

  // float *d_m, *d_v, *d_g, *d_p;

  // cudaMalloc((void**)&d_m, size_bytes);
  // cudaMalloc((void**)&d_v, size_bytes);
  // cudaMalloc((void**)&d_g, size_bytes);
  // cudaMalloc((void**)&d_p, size_bytes);
  
  // cudaMemcpy(d_m, m, size_bytes, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_v, v, size_bytes, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_g, g, size_bytes, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_p, p, size_bytes, cudaMemcpyHostToDevice);

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;

  const size_t threadsPerBlock = 256;
  const dim3 grids ((array_elements+threadsPerBlock-1) / threadsPerBlock);
  const dim3 blocks (threadsPerBlock);

  adamMode_t mode = ADAM_MODE_0;

  cudaDeviceSynchronize();

  double rts[repeat];
  for (int i = 0; i < repeat; i++) {
    auto start=std::chrono::steady_clock::now();
    adam<float, float><<<grids, blocks>>> (
      d_p, d_m, d_v, d_g,
      beta1, beta2,
      eps,
      grad_scale,
      step_size,
      time_step,
      array_elements,
      mode,
      decay);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time * 1e-9f;
  }
  printf("~ adam: %f (s)\n", rts[repeat / 2]);

  cudaFree(d_p);
  cudaFree(d_m);
  cudaFree(d_v);
  cudaFree(d_g);
  free(p);
  free(m);
  free(v);
  free(g);

  return 0;
}
