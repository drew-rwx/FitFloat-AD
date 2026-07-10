#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"

#include "../../FlexFloat.h"

__global__ 
void kernel1 (
    FFArr32 key, 
    FFArr32 query, 
    FFArr32 dot_product, 
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
    FFArr32 dot_product, 
    FFArr32 score, 
    const int n)
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;  
  if (i < n)
    score[i] = expf(dot_product[i]) / exp_sum[0];
}

__global__ 
void kernel3 (
    FFArr32 score, 
    FFArr32 value, 
    FFArr32 output, 
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
  FFArr32 d_key(n * d);
  FloatH2FFDmemcpy(d_key, key, n * d);

  FFArr32 d_value(n * d);
  FloatH2FFDmemcpy(d_value, value, n * d);

  FFArr32 d_query(d);
  FloatH2FFDmemcpy(d_query, query, d); 

  // intermediate
  FFArr32 d_dot_product(n);

  FFArr32 d_score(n);

  float *d_exp_sum;
  cudaMalloc((void**)&d_exp_sum, sizeof(float));

  // result
  float *output = (float*) malloc (d * sizeof(float));
  FFArr32 d_output(d);

  dim3 n_grid((n+255)/256);
  dim3 n_block(256);
  dim3 d_grid((d+255)/256);
  dim3 d_block(256);

  cudaDeviceSynchronize();

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

  FFD2FloatHmemcpy(output, d_output, d);
  cudaFree(d_exp_sum);
  return output;
}

int main(int argc, char* argv[]) {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  printf("%.2f GB (%zu Bytes) free, %.2f GB (%zu Bytes) total.\n", 1.0 * free_bytes / 1000000000, free_bytes, 1.0 * total_bytes / 1000000000, total_bytes);
  const size_t num_elements = total_bytes / 4 * 4 / 10 / 2048 / 2 / 32 * 32;
  printf("data has %zu elements!\n", num_elements);

  const size_t n = num_elements;
  const size_t d = 2048;
  const int r = 1;

  printf("Using %f GB\n", (1.0 * n*d*sizeof(float)*2 + d*sizeof(float)) / 1000000000);

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
