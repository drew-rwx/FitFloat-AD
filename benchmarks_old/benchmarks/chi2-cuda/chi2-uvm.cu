#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <random>
#include <cuda.h>

#include "../../FlexFloat.h"

__global__ void chi_kernel(
  const unsigned long long rows,
  const unsigned long long cols,
  const unsigned long long cRows,
  const unsigned long long contRows,
  const unsigned char *__restrict__ snpdata,
  float *__restrict__ results)
{
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= cols) return;

  unsigned char y;
  size_t m, n;
  unsigned int p = 0;
  int tot_cases = 1;
  int tot_controls= 1;
  int total = 1;
  float chisquare = 0.0f;
  float exp[3];        
  float Conexpected[3];        
  float Cexpected[3];
  float numerator1;
  float numerator2;

  int cases[3] = {1,1,1};
  int controls[3] = {1,1,1};

  // read cases: each thread reads a column of snpdata matrix
  for ( m = 0 ; m < cRows ; m++ ) {
    y = snpdata[(size_t)m * (size_t)cols + tid];
    if ( y == '0') { cases[0]++; }
    else if ( y == '1') { cases[1]++; }
    else if ( y == '2') { cases[2]++; }
  }

  // read controls: each thread reads a column of snpdata matrix
  for ( n = cRows ; n < cRows + contRows ; n++ ) {
    y = snpdata[(size_t)n * (size_t)cols + tid];
    if ( y == '0' ) { controls[0]++; }
    else if ( y == '1') { controls[1]++; }
    else if ( y == '2') { controls[2]++; }
  }

  for( p = 0 ; p < 3; p++ ) {
    tot_cases += cases[p];
    tot_controls += controls[p];
  }
  total = tot_cases + tot_controls;

  for( p = 0 ; p < 3; p++ ) {
    exp[p] = (float)cases[p] + controls[p]; 
    Cexpected[p] = tot_cases * exp[p] / total;
    Conexpected[p] = tot_controls * exp[p] / total;
    numerator1 = (float)cases[p] - Cexpected[p];
    numerator2 = (float)controls[p] - Conexpected[p];
    chisquare += numerator1 * numerator1 / Cexpected[p] +
                 numerator2 * numerator2 / Conexpected[p];
  }
  results[tid] = chisquare;
}

int main(int argc, char* argv[]) {

  const size_t rows = 4;

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
  const size_t array_elements = free_bytes * factor / (rows + 4);
  const size_t array_size_in_bytes = array_elements * 4;
  
  const long long native_difference = free_bytes - (array_elements * rows + array_size_in_bytes);
  printf("GPU memory: %zu free out of %zu total\n", free_bytes, total_bytes);
  printf("Native memory footprint: %zu bytes :: %lld byte difference from available memory\n", array_elements * rows + array_size_in_bytes, native_difference);

  size_t user_ff_bits_array[] = {32, 31, 28, 24, 20, 17};

  for (int i = 0; i < sizeof(user_ff_bits_array) / 8; i++) {
    const size_t user_ff_bits = user_ff_bits_array[i];
    const size_t ff_array_elements = get_ffarray_size_float(array_elements, user_ff_bits);
    const size_t ff_array_size_in_bytes = ff_array_elements * 4;

    const long long fitfloat_difference = (array_elements * rows + array_size_in_bytes) - (array_elements * rows + ff_array_size_in_bytes);
    const long long fitfloat_difference_available = free_bytes - (array_elements * rows + ff_array_size_in_bytes);
    printf("FF-%zu  memory footprint: %zu bytes :: %lld byte difference from native arrays, %lld from available memory\n", user_ff_bits, array_elements * rows + ff_array_size_in_bytes, fitfloat_difference, fitfloat_difference_available);    
  }

  size_t snpdata_size = sizeof(unsigned char) * rows * array_elements;
  size_t result_size = sizeof(float) * array_elements;

  printf("Using %zu bytes for snp and %zu bytes for result\n", snpdata_size, result_size);
  
  const size_t cols = array_elements;
  const size_t ncases = rows / 2;
  const size_t ncontrols = rows / 2;
  const int nthreads = 512;
  const int repeat = 9;

  // allocate SNP host data 

  unsigned char* dataT;
  cudaMallocManaged((void**) &dataT, snpdata_size);

  if(dataT == NULL) {
    printf("ERROR: Memory for data not allocated.\n");
    if (dataT) free(dataT);
    return 1;
  }

  std::mt19937 gen(19937); // mersenne_twister_engin
  std::uniform_int_distribution<> distrib(0, 2);
  for (size_t i = 0; i < snpdata_size; i++) {
    dataT[i] = distrib(gen) + '0';
  }

  // allocate SNP device data
  float* d_results;
  cudaMallocManaged((void**) &d_results, result_size);


  size_t jobs = cols;
  size_t nblocks = (jobs + nthreads - 1)/nthreads;

  cudaDeviceSynchronize();
  CheckCuda(__LINE__);
  

  double rts[repeat];
  for (int i = 0; i < repeat; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    chi_kernel <<< dim3(nblocks), dim3(nthreads) >>> (rows,cols,ncases,ncontrols,dataT,d_results);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time;
  }
  CheckCuda(__LINE__);
  cudaDeviceSynchronize();
  printf("~ Median chi_kernel execution time = %f (s)\n", (rts[repeat / 2]) * 1e-9f);


  cudaFree(dataT);
  cudaFree(d_results);
  CheckCuda(__LINE__);

  return 0;
}
