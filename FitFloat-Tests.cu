/*
This file is part of FitFloat, a floating-point array representation for GPUs that allows the user to choose the number of bits in the exponent and mantissa fields.
 
BSD 3-Clause License
 
Copyright (c) 2025, Andrew Rodriguez and Martin Burtscher
All rights reserved.
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
URL: The latest version of this code is available at https://github.com/burtscher/FitFloat.
 
Sponsor: This code is based upon work supported by the U.S. Department of Energy, National Nuclear Security Administration, under Award Number DE-NA0003969.
*/


#include <algorithm>
#include <cassert>

#include "FitFloat.h"

#ifndef TPB
  #define TPB 512
#endif


// ***********************
// ** Helpful Functions **
// ***********************


static void PrintGPUs()
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  for (int device = 0; device < deviceCount; ++device) {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, device);
      printf("Info: Device %d (%s) has compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
  }
}


template <typename T>
static T rand_fp(const T min, const T max)
{
  T scale = rand() / (T) RAND_MAX;    /* [0, 1.0f] */
  return min + scale * ( max - min ); /* [min, max] */
}


// *************
// ** Kernels **
// *************


template <typename T> // float*, FFArr32, FFarr64
__global__ void vector_init_with_value(T dest, const float val, const size_t num_elements)
{
  const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < num_elements)
  {
    dest[idx] = val;
  }
}

template <typename T> // float*, FFArr32, FFarr64
__global__ void vector_add(T a, T b, T c, const size_t num_elements)
{
  const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < num_elements) {
    c[idx] = a[idx] + b[idx];
  }
}

template <typename T> // float*, FFArr32, FFarr64
__global__ void vector_count(T a, unsigned long long* count, const size_t target_idx, const size_t block_size, const size_t num_elements)
{
  const auto target = a[target_idx];
  unsigned long long my_count = 0;

  const size_t s_idx = (blockDim.x * blockIdx.x + threadIdx.x) * block_size;
  const size_t limit = min(s_idx + block_size, num_elements);

  for (size_t idx = s_idx; idx < limit; idx++) {
    if ( a[idx] == target ) {
      my_count += 1;
    }
  }

  atomicAdd(count, my_count);
}


// *************************
// ** Experiments & Tests **
// *************************


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


template <typename T> // float, double
void test_correctness(const size_t num_elements, const bool use_special_case = false) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "test_correctness() can only be called with float and double types!");
  using FFArr = std::conditional_t<std::is_same<T, float>::value, FFArr32, FFArr64>;

  srand(0);

  if (use_special_case) {
    printf("*** %s Correctness (Special) ***\n\n", (std::is_same<T, float>::value) ? "Float" : "Double");
  } else {
    printf("*** %s Correctness (General, HtD Memcpy) ***\n\n", (std::is_same<T, float>::value) ? "Float" : "Double");
  }

  printf("%zu elements in each vector!\n", num_elements);

  const size_t max_vals_to_print = 0;

  const T min = 1.23456789f;
  const T max = 12345.6789f;

  // allocate init. array on host and device

  T* h_init = new T[num_elements];
  for (size_t idx = 0; idx < num_elements; idx++) {
    h_init[idx] = rand_fp<T>(min, max);
  }

  h_init[0] = 0.0f;
  h_init[1] = 1.0f;
  h_init[2] = -1.0f;
  h_init[3] = 2.0f;
  h_init[4] = 4.0f;

  T* d_init = nullptr;
  cudaMalloc((void **) &d_init, sizeof(T) * num_elements);
  cudaMemcpy(d_init, h_init, sizeof(T) * num_elements, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);

  GPUTimer d_timer;

  // allocate ff and init.

  FFArr ff(num_elements);
  CheckCuda(__LINE__);

  d_timer.start();
  if (use_special_case) {
    FF_initialize(ff, d_init, num_elements);
  } else {
    if constexpr (std::is_same<T, float>::value) {
      FloatH2FFDmemcpy(ff, h_init, num_elements);
    } else {
      DoubleH2FFDmemcpy(ff, h_init, num_elements);
    }
  }
  CheckCuda(__LINE__);
  printf("Init. time:: %6f s\n", d_timer.stop());


  // copy FF from device to host

  T* h_verify = new T[num_elements];
  if constexpr (std::is_same<T, float>::value) {
    FFD2FloatHmemcpy(h_verify, ff, num_elements);
  } else {
    FFD2DoubleHmemcpy(h_verify, ff, num_elements);
  }
  CheckCuda(__LINE__);

  // print values to compare

  for (size_t idx = 0; idx < std::min(max_vals_to_print, num_elements); idx++) {
    printf("            idx: %zu\n", idx);
    printf(" original value: %+f\n", h_init[idx]);
    printf("flexfloat value: %+f\n", h_verify[idx]);
    printf("     difference: %+f\n\n", h_init[idx] - h_verify[idx]);
  }
  printf("\n");

  // calculate error

  double ieee_sum = 0.0;
  double ff_sum = 0.0;
  double error_total = 0.0;

  for (size_t idx = 0; idx < num_elements; idx++) {
    ieee_sum += h_init[idx];
    ff_sum += h_verify[idx];
    error_total += h_init[idx] - h_verify[idx];
  }
  printf("Expected Sum: %f\n", ieee_sum);
  printf("  Actual Sum: %f\n", ff_sum);
  printf("  Diff Error: %f\n", ieee_sum - ff_sum);
  printf("   Sum Error: %f\n", error_total);

  // clean up

  delete [] h_init;
  cudaFree(d_init);

  delete [] h_verify;
}


template <typename T> // float, double
void test_correctness_with_file(const bool use_special_case = false) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "test_correctness_with_file() can only be called with float and double types!");
  
  using FFArr = std::conditional_t<std::is_same<T, float>::value, FFArr32, FFArr64>;
  const char * filename = (std::is_same<T, float>::value) ? "funnyFPvalues.fp32" : "funnyFPvalues.fp64" ;

  // header

  if (use_special_case) {
    printf("*** %s Correctness (Special) ***\n\n", (std::is_same<T, float>::value) ? "Float" : "Double");
  } else {
    printf("*** %s Correctness (General, HtD Memcpy) ***\n\n", (std::is_same<T, float>::value) ? "Float" : "Double");
  }

  // read in data from file

  FILE* const fin = fopen(filename, "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  const size_t num_elements = fsize / sizeof(T);

  // allocate init. array on host and device

  T* h_init = new T[num_elements];
  fseek(fin, 0, SEEK_SET);
  const long long insize = fread(h_init, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);

  T* d_init = nullptr;
  cudaMalloc((void **) &d_init, sizeof(T) * num_elements);
  cudaMemcpy(d_init, h_init, sizeof(T) * num_elements, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);

  GPUTimer d_timer;

  // allocate ff and init.

  FFArr ff(num_elements);
  CheckCuda(__LINE__);

  d_timer.start();
  if (use_special_case) {
    FF_initialize(ff, d_init, num_elements);
  } else {
    if constexpr (std::is_same<T, float>::value) {
      FloatH2FFDmemcpy(ff, h_init, num_elements);
    } else {
      DoubleH2FFDmemcpy(ff, h_init, num_elements);
    }
  }
  CheckCuda(__LINE__);
  printf("Init. time:: %6f s\n", d_timer.stop());

  // copy FF from device to host

  T* h_verify = new T[num_elements];
  if constexpr (std::is_same<T, float>::value) {
    FFD2FloatHmemcpy(h_verify, ff, num_elements);
  } else {
    FFD2DoubleHmemcpy(h_verify, ff, num_elements);
  }
  CheckCuda(__LINE__);

  // print values to compare

  for (size_t idx = 0; idx < num_elements; idx++) {
    printf("            idx: %zu\n", idx);
    printf(" original value: %+f\n", h_init[idx]);
    printf("flexfloat value: %+f\n", h_verify[idx]);
    printf("     difference: %+f\n\n", h_init[idx] - h_verify[idx]);
  }
  printf("\n");

  // clean up

  delete [] h_init;
  cudaFree(d_init);

  delete [] h_verify;
}


template <typename T> // float, double
void test_vector_add_FF(const size_t num_elements, const size_t num_runs, const bool use_special_case = false) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "test_vector_add_FF() can only be called with float and double types!");
  using FFArr = std::conditional_t<std::is_same<T, float>::value, FFArr32, FFArr64>;

  if (use_special_case) {
    printf("*** %s Vector Add (Special) ***\n\n", (std::is_same<T, float>::value) ? "Float" : "Double");
  } else {
    printf("*** %s Vector Add (General) ***\n\n", (std::is_same<T, float>::value) ? "Float" : "Double");
  }

  printf("%zu elements in each vector!\n", num_elements);

  const T init_value = 1.0f;

  // allocate ff a and ff b

  GPUTimer d_timer;
  double* runtimes = new double[num_runs];

  const int thread_blocks = (num_elements + TPB - 1) / TPB;

  FFArr ff_a(num_elements);
  CheckCuda(__LINE__);

  for (size_t r = 0; r < num_runs; r++) {
    if (use_special_case) {
      d_timer.start();

      FF_initialize(ff_a, init_value, num_elements);

      cudaDeviceSynchronize();
      runtimes[r] = d_timer.stop();
    } else {
      d_timer.start();

      vector_init_with_value<<<thread_blocks, TPB>>>(ff_a, init_value, num_elements);

      cudaDeviceSynchronize();
      runtimes[r] = d_timer.stop();
    }
    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### initialize a #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  FFArr ff_b(num_elements);
  CheckCuda(__LINE__);

  for (size_t r = 0; r < num_runs; r++) {
    if (use_special_case) {
      d_timer.start();

      FF_initialize(ff_b, init_value, num_elements);

      cudaDeviceSynchronize();
      runtimes[r] = d_timer.stop();
    } else {
      d_timer.start();

      vector_init_with_value<<<thread_blocks, TPB>>>(ff_b, init_value, num_elements);

      cudaDeviceSynchronize();
      runtimes[r] = d_timer.stop();
    }
    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### initialize b #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  // allocate ff c

  FFArr ff_c(num_elements);
  CheckCuda(__LINE__);

  // vector add

  for (size_t r = 0; r < num_runs; r++) {
    d_timer.start();

    vector_add<<<thread_blocks, TPB>>>(ff_a, ff_b, ff_c, num_elements);

    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();
    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### c = a + b #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  // clean up (ff handled in destructor)
  delete [] runtimes;
}


template <typename T> // float, double
void test_vector_add_IEEE(const size_t num_elements, const size_t num_runs) {
  printf("*** %s Vector Add (IEEE) ***\n\n", (std::is_same<T, float>::value) ? "Float" : "Double");
  printf("%zu elements in each vector!\n", num_elements);  

  const T init_value = 1.0f;

  // allocate a and b

  GPUTimer d_timer;
  double* runtimes = new double[num_runs];

  const int thread_blocks = (num_elements + TPB - 1) / TPB;

  T* d_a = nullptr;
  cudaMalloc((void **) &d_a, sizeof(T) * num_elements);
  CheckCuda(__LINE__);

  for (size_t r = 0; r < num_runs; r++) {
    d_timer.start();

    vector_init_with_value<<<thread_blocks, TPB>>>(d_a, init_value, num_elements);

    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();
    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### initialize a #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  T* d_b = nullptr;
  cudaMalloc((void **) &d_b, sizeof(T) * num_elements);
  CheckCuda(__LINE__);

  for (size_t r = 0; r < num_runs; r++) {
    d_timer.start();

    vector_init_with_value<<<thread_blocks, TPB>>>(d_b, init_value, num_elements);

    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();
    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### initialize b #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  // allocate ff c

  T* d_c = nullptr;
  cudaMalloc((void **) &d_c, sizeof(T) * num_elements);
  CheckCuda(__LINE__);

  // vector add

  for (size_t r = 0; r < num_runs; r++) {
    d_timer.start();

    vector_add<<<thread_blocks, TPB>>>(d_a, d_b, d_c, num_elements);

    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();
    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### c = a + b #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  // clean up

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  delete [] runtimes;
}


template <typename T> // float, double
void test_vector_count_FF(const size_t num_elements, const size_t num_runs) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "test_vector_add_FF() can only be called with float and double types!");
  using FFArr = std::conditional_t<std::is_same<T, float>::value, FFArr32, FFArr64>;

  printf("*** %s Vector Count ***\n\n", (std::is_same<T, float>::value) ? "Float" : "Double");
  printf("%zu elements in each vector!\n", num_elements);

  // allocate init on host

  unsigned long long expected_count = 0;

  T* h_init = new T[num_elements];
  #pragma omp parallel for default(none) shared(h_init, num_elements) reduction(+:expected_count)
  for (size_t idx = 0; idx < num_elements; idx++) {
    T val = 1.0f;

    if (idx % 2 == 0) {
      val = 0.0f;
    }

    h_init[idx] = val;

    if (val == h_init[0]) {
      expected_count++;
    }
  }

  // allocate FF a

  FFArr ff_a(num_elements);
  CheckCuda(__LINE__);

  GPUTimer d_timer;
  double* runtimes = new double[num_elements];

  // memcpy

  for (size_t r = 0; r < num_runs; r++) {
    d_timer.start();

    if constexpr (std::is_same<T, float>::value) {
      FloatH2FFDmemcpy(ff_a, h_init, num_elements);
    } else {
      DoubleH2FFDmemcpy(ff_a, h_init, num_elements);
    }

    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();

    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### memcpy a #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  // count a

  const size_t block_size = 32; // how many elements each thread will look at
  const size_t num_blocks = (num_elements + block_size - 1) / block_size;

  const size_t count_thread_blocks = (num_blocks + TPB - 1) / TPB;

  unsigned long long* d_count = nullptr;
  cudaMalloc((void **) &d_count, sizeof(unsigned long long));

  for (size_t r = 0; r < num_runs; r++) {
    cudaMemset(d_count, 0, sizeof(unsigned long long)); // set the counter to zero

    d_timer.start();

    vector_count<<<count_thread_blocks, TPB>>>(ff_a, d_count, 0, block_size, num_elements);

    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();
    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### count a #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  // memcpy dev to host

  for (size_t r = 0; r < num_runs; r++) {
    d_timer.start();

    if constexpr (std::is_same<T, float>::value) {
      FFD2FloatHmemcpy(h_init, ff_a, num_elements);
    } else {
      FFD2DoubleHmemcpy(h_init, ff_a, num_elements);
    }

    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();

    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### memcpy a (dev to host) #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  // verify
  unsigned long long h_count = 0;
  cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  printf("Expected count: %llu\n", expected_count);
  printf("  Actual count: %llu\n", h_count);

  // clean up (ff handled in destructor)
  cudaFree(d_count);
  delete [] h_init;
  delete [] runtimes;
}


template <typename T> // float, double
void test_vector_count_IEEE(const size_t num_elements, const size_t num_runs) {
  printf("*** %s Vector Count (IEEE) ***\n\n", (std::is_same<T, float>::value) ? "Float" : "Double");
  printf("%zu elements in each vector!\n", num_elements);  

  // allocate init on host

  unsigned long long expected_count = 0;

  T* h_init = new T[num_elements];
  #pragma omp parallel for default(none) shared(h_init, num_elements) reduction(+:expected_count)
  for (size_t idx = 0; idx < num_elements; idx++) {
    T val = 1.0f;

    if (idx % 2 == 0) {
      val = 0.0f;
    }

    h_init[idx] = val;

    if (val == h_init[0]) {
      expected_count++;
    }
  }

  // allocate a on device

  T* d_a = nullptr;
  cudaMalloc((void **) &d_a, sizeof(T) * num_elements);
  CheckCuda(__LINE__);

  GPUTimer d_timer;
  double* runtimes = new double[num_elements];

  // memcpy

  for (size_t r = 0; r < num_runs; r++) {
    d_timer.start();

    cudaMemcpy(d_a, h_init, sizeof(T) * num_elements, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();

    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### memcpy a #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  // count a

  const size_t block_size = 32; // how many elements each thread will look at
  const size_t num_blocks = (num_elements + block_size - 1) / block_size;

  const size_t count_thread_blocks = (num_blocks + TPB - 1) / TPB;

  unsigned long long* d_count = nullptr;
  cudaMalloc((void **) &d_count, sizeof(unsigned long long));

  for (size_t r = 0; r < num_runs; r++) {
    cudaMemset(d_count, 0, sizeof(unsigned long long)); // set the counter to zero

    d_timer.start();

    vector_count<<<count_thread_blocks, TPB>>>(d_a, d_count, 0, block_size, num_elements);

    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();
    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### count a #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  // memcpy dev to host

  for (size_t r = 0; r < num_runs; r++) {
    d_timer.start();

    cudaMemcpy(h_init, d_a, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    runtimes[r] = d_timer.stop();

    CheckCuda(__LINE__);
  }

  // report performance

  std::sort(runtimes, runtimes + num_runs);
  printf("##### memcpy a (dev to host) #####\n");
  printf("   min: %6f s\n", runtimes[0]);
  printf("   max: %6f s\n", runtimes[num_runs - 1]);
  printf("median: %6f s\n", runtimes[num_runs / 2]);

  // verify
  unsigned long long h_count = 0;
  cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  printf("Expected count: %llu\n", expected_count);
  printf("  Actual count: %llu\n", h_count);

  // clean up (ff handled in destructor)
  cudaFree(d_a);
  cudaFree(d_count);
  delete [] h_init;
  delete [] runtimes;
}


// **********
// ** Main **
// **********


int main(int argc, char* argv [])
{
  if (argc != 3) { printf("USAGE: num_runs test_number\n"); exit(-1); }
  const size_t num_runs = atol(argv[1]);
  const size_t test_number = atol(argv[2]);

  printf("FlexFloat v1.0 (%s)\n", __FILE__);
  printf("GPU version\n");
  printf("Copyright 2025 Texas State University\n\n");

  PrintGPUs();
  printf("\n");

  // query memory

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  CheckCuda(__LINE__);

  printf("%.2f GB (%zu Bytes) free, %.2f GB (%zu Bytes) total.\n", 1.0 * free_bytes / 1000000000, free_bytes, 1.0 * total_bytes / 1000000000, total_bytes);

  // experiments and tests

  switch (test_number) {

    case 20: { // test_correctness()
      const size_t num_elements_64 = free_bytes / 8 * 8 / 10 / 2 / 32 * 32;
      test_correctness<double>(num_elements_64);
      test_correctness<double>(num_elements_64, true);

      break;
    }

    case 0: { // test_correctness()
      const size_t num_elements_32 = free_bytes / 4 * 8 / 10 / 2 / 32 * 32;
      test_correctness<float>(num_elements_32);
      test_correctness<float>(num_elements_32, true);

      break;
    }

    case 1: { // test_correctness_with_file()
      test_correctness_with_file<float>();
      test_correctness_with_file<float>(true);
      test_correctness_with_file<double>();
      test_correctness_with_file<double>(true);

      break;
    }

    case 2: { // test_vector_add_FF<float>()
      const size_t num_elements_32 = free_bytes / 4 * 8 / 10 / 3 / 32 * 32; // consistent

      test_vector_add_FF<float>(num_elements_32, num_runs);

      break;
    }

    case 3: { // test_vector_add_FF<float>(special)
      const size_t num_elements_32 = free_bytes / 4 * 8 / 10 / 3 / 32 * 32; // consistent

      test_vector_add_FF<float>(num_elements_32, num_runs, true);

      break;
    }

    case 4: { // test_vector_add_IEEE<float>()
      const size_t num_elements_32 = free_bytes / 4 * 8 / 10 / 3 / 32 * 32; // consistent

      test_vector_add_IEEE<float>(num_elements_32, num_runs);

      break;
    }

    case 5: { // test_vector_add_FF<double>()
      const size_t num_elements_64 = free_bytes / 8 * 8 / 10 / 3 / 32 * 32; // consistent

      test_vector_add_FF<double>(num_elements_64, num_runs);

      break;
    }

    case 6: { // test_vector_add_FF<double>(special)
      const size_t num_elements_64 = free_bytes / 8 * 8 / 10 / 3 / 32 * 32; // consistent

      test_vector_add_FF<double>(num_elements_64, num_runs, true);

      break;
    }

    case 7: { // test_vector_add_IEEE<double>()
      const size_t num_elements_64 = free_bytes / 8 * 8 / 10 / 3 / 32 * 32; // consistent

      test_vector_add_IEEE<double>(num_elements_64, num_runs);

      break;
    }

    case 8: { // test_vector_count_FF<float>()
      const size_t num_elements_32 = free_bytes / 4 * 8 / 10 / 32 * 32; // consistent

      test_vector_count_FF<float>(num_elements_32, num_runs);

      break;
    }

    case 9: { // test_vector_count_IEEE<float>()
      const size_t num_elements_32 = free_bytes / 4 * 8 / 10 / 32 * 32; // consistent

      test_vector_count_IEEE<float>(num_elements_32, num_runs);

      break;
    }

    case 10: { // test_vector_count_FF<double>()
      const size_t num_elements_64 = free_bytes / 8 * 8 / 10 / 32 * 32; // consistent

      test_vector_count_FF<double>(num_elements_64, num_runs);

      break;
    }

    case 11: { // test_vector_count_IEEE<double>()
      const size_t num_elements_64 = free_bytes / 8 * 8 / 10 / 32 * 32; // consistent

      test_vector_count_IEEE<double>(num_elements_64, num_runs);

      break;
    }

    default: {
      printf("ERR: Invalid test number!\n");
      break;
    }

  }

  return 0;
}
