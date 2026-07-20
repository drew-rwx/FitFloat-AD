#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
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
void kernel1 (
    const float*__restrict__ key, 
    const float*__restrict__ query, 
    float*__restrict__ dot_product, 
    float*__restrict__ exp_sum, 
    const int n,
    const int d) 
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;  
  if (i < n) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    atomicAdd(exp_sum, expf(sum));
  }
}

__global__ 
void kernel2 (
    const float*__restrict__ exp_sum, 
    const float*__restrict__ dot_product, 
    float*__restrict__ score, 
    const int n)
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;  
  if (i < n)
    score[i] = expf(dot_product[i]) / exp_sum[0];
}

__global__ 
void kernel3 (
    const float*__restrict__ score, 
    const float*__restrict__ value, 
    float*__restrict__ output, 
    const int n,
    const int d) 
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;  
  if (j < d) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}

float* attention_device(const float* key, const float* value, const float* query,
                        const size_t n, const size_t d, const int repeat) 
{
  // input
  float *d_key;
  cudaMallocManaged((void**)&d_key, n * d * sizeof(float)); 
  cudaMemcpy(d_key, key, n * d * sizeof(float), cudaMemcpyHostToHost); 

  float *d_value;
  cudaMallocManaged((void**)&d_value, n * d * sizeof(float)); 
  cudaMemcpy(d_value, value, n * d * sizeof(float), cudaMemcpyHostToHost); 

  float *d_query;
  cudaMallocManaged((void**)&d_query, d * sizeof(float)); 
  cudaMemcpy(d_query, query, d * sizeof(float), cudaMemcpyHostToHost); 

  // intermediate
  float *d_dot_product;
  cudaMallocManaged((void**)&d_dot_product, n * sizeof(float));

  float *d_score;
  cudaMallocManaged((void**)&d_score, n * sizeof(float));

  float *d_exp_sum;
  cudaMalloc((void**)&d_exp_sum, sizeof(float));

  // result
  float *output = (float*) malloc (d * sizeof(float));
  float *d_output;
  cudaMallocManaged((void**)&d_output, d * sizeof(float));

  dim3 n_grid((n+255)/256);
  dim3 n_block(256);
  dim3 d_grid((d+255)/256);
  dim3 d_block(256);

  cudaDeviceSynchronize();
  CheckCuda(__LINE__);

  double rts[repeat];
  for (int k = 0; k < repeat; k++) {
    cudaMemset(d_exp_sum, 0, 4);
    cudaDeviceSynchronize();
    auto start=std::chrono::steady_clock::now();
    kernel1<<<n_grid, n_block>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);

    kernel2<<<n_grid, n_block>>>(d_exp_sum, d_dot_product, d_score, n);

    kernel3<<<d_grid, d_block>>>(d_score, d_value, d_output, n, d);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[k] = time * 1e-9f;
  }
  printf("~ attention: %f (s)\n", rts[repeat / 2]);
  CheckCuda(__LINE__);

  


  // cudaMemcpy(output, d_output, d * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_score);
  cudaFree(d_value);
  cudaFree(d_output);
  cudaFree(d_key);
  cudaFree(d_dot_product);
  cudaFree(d_exp_sum);
  return output;
}

int main(int argc, char* argv[]) {
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
  free_bytes -= 2048 * 4 * 2;

  const size_t array_elements = free_bytes * factor / 4 / 2; // 2 arrays

  const size_t n = array_elements / 2048;
  const size_t d = 2048;
  const int r = 3;

  // input
  float* key = (float*) malloc (n * d * sizeof(float));
  float* value = (float*) malloc (n * d * sizeof(float));
  float* query = (float*) malloc (d * sizeof(float));

  srand(2);
  for (int i = 0; i < n * d; i++) {
    key[i] = 0.1;
    value[i] = 0.3;
    if (rand() % 2)
      query[i % d] = value[i] + key[i] ;
    else
      query[i % d] = value[i] - key[i] ;
  }

  float* dout = attention_device(key, value, query, n, d, r);

  free(key);
  free(value);
  free(query);
  free(dout);
  return 0;
}
