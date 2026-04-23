/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <algorithm>  // for_each
#include <chrono>
#include <cuda.h>

#include "support/common.h"
#include "support/verify.h"
#include "../../FitFloat.h"

#if defined(UVM1)
  #define factor 1.1
#else
  #define factor 2.5
#endif

// GPU kernel 
__global__ void PTTWAC_soa_asta(const int A, 
                                const int B, 
                                const int b, 
                                  astaT *__restrict__ input, 
                                int *__restrict__ finished, 
                                int *__restrict__ head) 
{
  __shared__ int lmem[2];

  const int tid = threadIdx.x;
  int       m   = A * B - 1;

  if(tid == 0) // Dynamic fetch
    lmem[1] = atomicAdd(&head[0], 1);
  __syncthreads();

  while(lmem[1] < m) {
    int next_in_cycle = (lmem[1] * A) - m * (lmem[1] / B);
    if(next_in_cycle == lmem[1]) {
      if(tid == 0) // Dynamic fetch
        lmem[1] = atomicAdd(&head[0], 1);
      __syncthreads();
      continue;
    }
    astaT   data1, data2, data3, data4;
    int i = tid;
    if(i < b)
      data1 = input[lmem[1] * b + i];
    i += blockDim.x;
    if(i < b)
      data2 = input[lmem[1] * b + i];
    i += blockDim.x;
    if(i < b)
      data3 = input[lmem[1] * b + i];
    i += blockDim.x;
    if(i < b)
      data4 = input[lmem[1] * b + i];

    if(tid == 0) {
      //make sure the read is not cached
      lmem[0] = atomicAdd(&finished[lmem[1]], 0);
    }
    __syncthreads();

    for(; lmem[0] == 0; next_in_cycle = (next_in_cycle * A) - m * (next_in_cycle / B)) {
      astaT backup1, backup2, backup3, backup4;
      i = tid;
      if(i < b)
        backup1 = input[next_in_cycle * b + i];
      i += blockDim.x;
      if(i < b)
        backup2 = input[next_in_cycle * b + i];
      i += blockDim.x;
      if(i < b)
        backup3 = input[next_in_cycle * b + i];
      i += blockDim.x;
      if(i < b)
        backup4 = input[next_in_cycle * b + i];

      if(tid == 0) {
        lmem[0] = atomicExch(&finished[next_in_cycle], (int)1);
      }
      __syncthreads();

      if(!lmem[0]) {
        i = tid;
        if(i < b)
          input[next_in_cycle * b + i] = data1;
        i += blockDim.x;
        if(i < b)
          input[next_in_cycle * b + i] = data2;
        i += blockDim.x;
        if(i < b)
          input[next_in_cycle * b + i] = data3;
        i += blockDim.x;
        if(i < b)
          input[next_in_cycle * b + i] = data4;
      }
      i = tid;
      if(i < b)
        data1 = backup1;
      i += blockDim.x;
      if(i < b)
        data2 = backup2;
      i += blockDim.x;
      if(i < b)
        data3 = backup3;
      i += blockDim.x;
      if(i < b)
        data4 = backup4;
    }

    if(tid == 0) // Dynamic fetch
      lmem[1] = atomicAdd(&head[0], 1);
    __syncthreads();
  }
}


// Params ---------------------------------------------------------------------
struct Params {

  int device;
  int n_gpu_threads;
  int n_gpu_blocks;
  int n_warmup;
  int n_reps;
  int m;
  size_t n;
  int s;

  Params(int argc, char **argv) {
    device        = 0;
    n_gpu_threads = 256;
    n_gpu_blocks  = 32;
    n_warmup      = 0;
    n_reps        = 3;
    m             = 197;
    n             = 35588;
    s             = 32;
    int opt;
    while((opt = getopt(argc, argv, "hd:i:g:w:r:m:n:s:")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'i': n_gpu_threads = atoi(optarg); break;
        case 'g': n_gpu_blocks  = atoi(optarg); break;
        case 'w': n_warmup      = atoi(optarg); break;
        case 'r': n_reps        = atoi(optarg); break;
        case 'm': m             = atoi(optarg); break;
        case 'n': n             = atoi(optarg); break;
        case 's': s             = atoi(optarg); break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
      }
    }
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./trns [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -i <I>    # of device threads per block (default=64)"
        "\n    -g <G>    # of device blocks (default=16)"
        "\n    -w <W>    # of untimed warmup iterations (default=10)"
        "\n    -r <R>    # of timed repetition iterations (default=100)"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -m <M>    matrix height (default=197)"
        "\n    -n <N>    matrix width (default=35588)"
        "\n    -s <M>    super-element size (default=32)"
        "\n");
  }
};

// Input Data -----------------------------------------------------------------
void read_input(astaT *x_vector, const Params &p) {
  int tiled_n = divceil(p.n, p.s);
  int in_size = p.m * tiled_n * p.s;
  srand(5432);
  for(int i = 0; i < in_size; i++) {
    x_vector[i] = ((astaT)(rand() % 100) / 100);
  }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
  Params p(argc, argv);
  int blocks = p.n_gpu_blocks;
  int threads = p.n_gpu_threads;
  const int max_gpu_threads = 256;
  assert(threads <= max_gpu_threads && 
         "The thread block size is at most 256");

  // query memory

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);

  size_t reserved_size = free_bytes - 70000000;

  int* reserved = nullptr;
  cudaMalloc((void**) &reserved, reserved_size);
  cudaMemset(reserved, 1, reserved_size);
  cudaDeviceSynchronize();

  cudaMemGetInfo(&free_bytes, &total_bytes);

  const size_t array_elements = free_bytes * factor / 8 / 2; // 2 arrays

  p.n = array_elements / 256;
  p.m = 256;
  p.s = 32;

  // Allocate
  size_t tiled_n       = divceil(p.n, p.s);
  size_t in_size       = p.m * tiled_n * p.s;
  size_t finished_size = p.m * tiled_n;

  size_t in_size_bytes = in_size * sizeof(astaT);
  size_t finished_size_bytes = finished_size * sizeof(int);

  printf("Using %.2f GB for data\n", 1.0 * (in_size_bytes + finished_size_bytes) / 1000000000);

  astaT *h_in_out = (astaT *)malloc(in_size_bytes);
  int *h_finished = (int *)malloc(finished_size_bytes);
  int *h_head = (int *)malloc(sizeof(int));


  dim3 dimGrid(blocks);
  dim3 dimBlock(threads);

  astaT * d_in_out;
  int * d_finished;
  int * d_head;
  cudaMallocManaged((void**)&d_in_out, in_size_bytes);
  cudaMallocManaged((void**)&d_finished, finished_size_bytes);
  cudaMalloc((void**)&d_head, sizeof(int));
  astaT *h_in_backup = (astaT *)malloc(in_size_bytes);

  // Initialize
  read_input(h_in_out, p);
  memset((void *)h_finished, 0, finished_size_bytes);
  h_head[0] = 0;
  memcpy(h_in_backup, h_in_out, in_size_bytes); // Backup for reuse across iterations

  double time = 0;

  // Loop over the kernel on a device
  double rts[3];
  for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

    cudaMemcpyAsync(d_in_out, h_in_backup, in_size_bytes, cudaMemcpyHostToHost, 0);
    cudaMemcpyAsync(d_finished, h_finished, finished_size_bytes, cudaMemcpyHostToHost, 0);
    cudaMemcpyAsync(d_head, h_head, sizeof(int), cudaMemcpyHostToDevice, 0);

    cudaDeviceSynchronize();
    auto start=std::chrono::steady_clock::now();
    PTTWAC_soa_asta<<<dimGrid, dimBlock>>>(p.m, tiled_n, p.s, d_in_out, d_finished, d_head);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[rep] = time * 1e-9f;
  }
  printf("~ asta: %f (s)\n", rts[1]);


  // Free memory
  free(h_in_out);
  free(h_finished);
  free(h_head);
  free(h_in_backup);

  cudaFree(d_in_out);
  cudaFree(d_finished);
  cudaFree(d_head);

  return 0;
}
