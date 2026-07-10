#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <cuda.h>
#include "reference.h"

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
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  printf("%.2f GB (%zu Bytes) free, %.2f GB (%zu Bytes) total.\n", 1.0 * free_bytes / 1000000000, free_bytes, 1.0 * total_bytes / 1000000000, total_bytes);
  total_bytes = ((size_t) 1024) * 1024 * 1024 * 3;
  const size_t num_elements = total_bytes / 4 * 9 / 10 / 4 / 32 * 32;
  printf("data has %zu elements!\n", num_elements);

  const size_t vector_size = num_elements;
  const int time_step = 200;
  const int repeat = 9;

  size_t size_bytes = vector_size * sizeof(float);

  printf("Using %.2f GB for data\n", 4 * 1.0 * size_bytes / 1000000000);

  float *m = (float*) malloc (size_bytes);
  float *v = (float*) malloc (size_bytes);
  float *g = (float*) malloc (size_bytes);
  float *p = (float*) malloc (size_bytes);
  float *r = (float*) malloc (size_bytes);

  srand(123);
  for (int i = 0; i < vector_size; i++) {
    m[i] = rand() / (float)RAND_MAX;
    v[i] = rand() / (float)RAND_MAX;
    g[i] = rand() / (float)RAND_MAX;
    r[i] = p[i] = rand() / (float)RAND_MAX;
  }

  float *d_m, *d_v, *d_g, *d_p;

  cudaMalloc((void**)&d_m, size_bytes);
  cudaMalloc((void**)&d_v, size_bytes);
  cudaMalloc((void**)&d_g, size_bytes);
  cudaMalloc((void**)&d_p, size_bytes);
  
  cudaMemcpy(d_m, m, size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_g, g, size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_p, p, size_bytes, cudaMemcpyHostToDevice);

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;

  const int threadsPerBlock = 256;
  const dim3 grids ((vector_size+threadsPerBlock-1) / threadsPerBlock);
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
      vector_size,
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
  free(r);
  return 0;
}
