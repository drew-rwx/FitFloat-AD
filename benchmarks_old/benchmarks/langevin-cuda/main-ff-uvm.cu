#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

#include "../../FlexFloat.h"

__global__
void k0 (FFArr32 a, FFArr32 o) {
  size_t t = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[t];
  o[t] = coshf(x)/sinhf(x) - 1.f/x;
}

__global__
void k1 (FFArr32 a, FFArr32 o) {
  size_t t = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[t];
  o[t] = 1.f / tanhf(x) - 1.f/x;
}

/*
Copyright (c) 2018-2021, Norbert Juffa
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

__global__
void k2 (FFArr32 a, FFArr32 o) {
  size_t t = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[t];
  float s, r;
  s = x * x;
  r =              7.70960469e-8f;
  r = fmaf (r, s, -1.65101926e-6f);
  r = fmaf (r, s,  2.03457112e-5f);
  r = fmaf (r, s, -2.10521728e-4f);
  r = fmaf (r, s,  2.11580913e-3f);
  r = fmaf (r, s, -2.22220998e-2f);
  r = fmaf (r, s,  8.33333284e-2f);
  r = fmaf (r, x,  0.25f * x);
  o[t] = r;
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

  const int repeat = 9;



  float *a = nullptr;

  a = (float*) malloc (array_size_in_bytes);
  // the range [-1.8, -0.00001)
  for (size_t i = 0; i < array_elements; i++) {
    a[i] = -1.8f + i * (1.79999f / array_elements);
  }

  FFArr32 d_o0(array_elements);
  FFArr32 d_o1(array_elements);
  FFArr32 d_o2(array_elements);

  FFArr32 d_a(array_elements);
  FloatH2FFDmemcpy(d_a, a, array_elements);

  cudaDeviceSynchronize();
  double rts[repeat];
  
  for (int i = 0; i < repeat; i++) {
    auto start = std::chrono::steady_clock::now();
    k0<<<array_elements/256, 256>>>(d_a, d_o0);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time;
  }
  printf("~ Median execution time of k0: %f (s)\n", (rts[repeat / 2] * 1e-9f));

  for (int i = 0; i < repeat; i++) {
    auto start = std::chrono::steady_clock::now();
    k1<<<array_elements/256, 256>>>(d_a, d_o1);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time;
  }
  printf("~ Median execution time of k1: %f (s)\n", (rts[repeat / 2] * 1e-9f));

  for (int i = 0; i < repeat; i++) {
    auto start = std::chrono::steady_clock::now();
    k2<<<array_elements/256, 256>>>(d_a, d_o2);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time;
  }
  printf("~ Median execution time of k2: %f (s)\n", (rts[repeat / 2] * 1e-9f));

  free(a);
  // cudaFree(d_a);
  // cudaFree(d_o0);
  // cudaFree(d_o1);
  // cudaFree(d_o2);
  return 0;
}
