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

#if defined(INPUT_SIZE_SMALL)
  #define reserve_gb 2
#else
  #define reserve_gb 8
#endif

__global__
void adam (
        FFArr32 p,
        FFArr32 m,
        FFArr32 v,
        FFArr32 g,
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
    float mj = m[j];
    float vj = v[j];
    float gj = g[j];
    float pj = p[j];
    for (int t = 0; t < time_step; t++) {
      float scaled_grad = gj/grad_scale;
      mj = b1*mj + (1.f-b1)*scaled_grad;
      vj = b2*vj + (1.f-b2)*scaled_grad*scaled_grad;
      float m_corrected = mj / (1.f-powf(b1, t));
      float v_corrected = vj / (1.f-powf(b2, t));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sqrtf(v_corrected + eps);
      else // Mode 1
        denom = sqrtf(v_corrected) + eps;
      float update = (m_corrected/denom) + (decay*pj);
      pj -= (step_size*update);
    }
    m[j] = mj;
    v[j] = vj;
    p[j] = pj;
  }
}

int main(int argc, char* argv[])
{
  // query memory

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);

  size_t reserved_size = free_bytes - ((size_t) 1024) * 1024 * 1024 * reserve_gb;

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

  FFArr32 d_m(array_elements);
  FFArr32 d_v(array_elements);
  FFArr32 d_g(array_elements);
  FFArr32 d_p(array_elements);

  FloatH2FFDmemcpy(d_m, m, array_elements);
  FloatH2FFDmemcpy(d_v, v, array_elements);
  FloatH2FFDmemcpy(d_g, g, array_elements);
  FloatH2FFDmemcpy(d_p, p, array_elements);

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
    adam<<<grids, blocks>>> (
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

  free(p);
  free(m);
  free(v);
  free(g);
  return 0;
}
