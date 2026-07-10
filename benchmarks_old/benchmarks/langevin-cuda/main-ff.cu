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

int main(int argc, char* argv[]) {

  const size_t n = 4ll * 1000000000 / 4 / 4;
  const int repeat = 1;
  const size_t size = sizeof(float) * n;

  float *a, *o, *o0, *o1, *o2;

  a = (float*) malloc (size);
  o = (float*) malloc (size);
  // the range [-1.8, -0.00001)
  for (int i = 0; i < n; i++) {
    a[i] = -1.8f + i * (1.79999f / n);
  }

  o0 = (float*) malloc (size);
  o1 = (float*) malloc (size);
  o2 = (float*) malloc (size);

  // float *d_o0, *d_o1, *d_o2;
  // // cudaMalloc((void**)&d_a, size);
  // cudaMalloc((void**)&d_o0, size);
  // cudaMalloc((void**)&d_o1, size);
  // cudaMalloc((void**)&d_o2, size);

  FFArr32 d_o0(size / 4);
  FFArr32 d_o1(size / 4);
  FFArr32 d_o2(size / 4);

  FFArr32 d_a(size / 4);
  FloatH2FFDmemcpy(d_a, a, size / 4);

  // cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    k0<<<n/256, 256>>>(d_a, d_o0);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of k0: %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    k1<<<n/256, 256>>>(d_a, d_o1);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of k1: %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    k2<<<n/256, 256>>>(d_a, d_o2);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of k2: %f (s)\n", (time * 1e-9f) / repeat);

  free(a);
  free(o);
  free(o0);
  free(o1);
  free(o2);
  // cudaFree(d_a);
  // cudaFree(d_o0);
  // cudaFree(d_o1);
  // cudaFree(d_o2);
  return 0;
}
