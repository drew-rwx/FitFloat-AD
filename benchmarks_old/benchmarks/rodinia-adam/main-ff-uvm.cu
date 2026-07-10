#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <cuda.h>
#include "reference.h"
#include "../../FlexFloat.h"

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
    for (int t = 0; t < time_step; t++) {
      float scaled_grad = g[j]/grad_scale;
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
      p[j] = p[j] - (step_size*update);
    }
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

  const double factor = 3.0;
  const size_t array_elements = free_bytes * factor / 4 / 4; // four arrays
  const size_t array_size_in_bytes = array_elements * 4;
  
  const long long native_difference = free_bytes - array_size_in_bytes * 4;
  printf("GPU memory: %zu free out of %zu total\n", free_bytes, total_bytes);
  printf("Native memory footprint: %zu bytes :: %lld byte difference from available memory\n", array_size_in_bytes * 4, native_difference);

  size_t user_ff_bits_array[] = {32, 31, 28, 24, 20, 17};

  for (int i = 0; i < sizeof(user_ff_bits_array) / 8; i++) {
    const size_t user_ff_bits = user_ff_bits_array[i];
    const size_t ff_array_elements = get_ffarray_size_float(array_elements, user_ff_bits);
    const size_t ff_array_size_in_bytes = ff_array_elements * 4;

    const long long fitfloat_difference = array_size_in_bytes * 4 - ff_array_size_in_bytes * 4;
    const long long fitfloat_difference_available = free_bytes - ff_array_size_in_bytes * 4;
    printf("FF-%zu  memory footprint: %zu bytes :: %lld byte difference from native arrays, %lld from available memory\n", user_ff_bits, ff_array_size_in_bytes * 4, fitfloat_difference, fitfloat_difference_available);    
  }

  const int time_step = 200;
  const int repeat = 9;

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
