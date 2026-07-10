/*
 * GPU-accelerated AIDW interpolation algorithm 
 *
 * Implemented with / without Shared Memory
 *
 * By Dr.Gang Mei
 *
 * Created on 2015.11.06, China University of Geosciences, 
 *                        gang.mei@cugb.edu.cn
 * Revised on 2015.12.14, China University of Geosciences, 
 *                        gang.mei@cugb.edu.cn
 * 
 * Related publications:
 *  1) "Evaluating the Power of GPU Acceleration for IDW Interpolation Algorithm"
 *     http://www.hindawi.com/journals/tswj/2014/171574/
 *  2) "Accelerating Adaptive IDW Interpolation Algorithm on a Single GPU"
 *     http://arxiv.org/abs/1511.02186
 *
 * License: http://creativecommons.org/licenses/by/4.0/
 */

#include <cstdio>
#include <cstdlib>     
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda.h>
#include "reference.h"
 #include "../../FlexFloat.h"

// Calculate the power parameter, and then weighted interpolating
// Without using shared memory
__global__
void AIDW_Kernel(
    const float *__restrict dx, 
    const float *__restrict dy,
    const float *__restrict dz,
    const int dnum,
    const float *__restrict ix,
    const float *__restrict iy,
          float *__restrict iz,
    const int inum,
    const float area,
    const float *__restrict avg_dist) 

{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < inum) {
    float sum = 0.f, dist = 0.f, t = 0.f, z = 0.f, alpha = 1.f;

    float r_obs = avg_dist[tid];                // The observed average nearest neighbor distance
    float r_exp = 0.5f / sqrtf(dnum / area);    // The expected nearest neighbor distance for a random pattern
    float R_S0 = r_obs / r_exp;                 // The nearest neighbor statistic

    // Normalize the R(S0) measure such that it is bounded by 0 and 1 by a fuzzy membership function 
    float u_R = 0.f;
    if(R_S0 >= R_min) u_R = 0.5f-0.5f * cosf(3.1415926f / R_max * (R_S0 - R_min));
    if(R_S0 >= R_max) u_R = 1.f;

    // Determine the appropriate distance-decay parameter alpha by a triangular membership function
    // Adaptive power parameter: a (alpha)
    if(u_R>= 0.f && u_R<=0.1f)  alpha = a1; 
    if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
    if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
    if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
    if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
    if(u_R>0.9f && u_R<=1.f)  alpha = a5;
    alpha *= 0.5f; // Half of the power

    // Weighted average
    for(int j = 0; j < dnum; j++) {
      dist = (ix[tid] - dx[j]) * (ix[tid] - dx[j]) + (iy[tid] - dy[j]) * (iy[tid] - dy[j]) ;
      t = 1.f / powf(dist, alpha);  sum += t;  z += dz[j] * t;
    }
    iz[tid] = z / sum;
  }
}

// Calculate the power parameter, and then weighted interpolating
// With using shared memory (Tiled version of the stage 2)
__global__
void AIDW_Kernel_Tiled(
    const float *__restrict dx, 
    const float *__restrict dy,
    const float *__restrict dz,
    const size_t dnum,
    const float *__restrict ix,
    const float *__restrict iy,
          float *__restrict iz,
    const size_t inum,
    const float area,
    const float *__restrict avg_dist)
{
  // Shared Memory
  __shared__ float sdx[BLOCK_SIZE];
  __shared__ float sdy[BLOCK_SIZE];
  __shared__ float sdz[BLOCK_SIZE];

  size_t tid = threadIdx.x + blockIdx.x * blockDim.x; 
  if (tid >= inum) return;

  float dist = 0.f, t = 0.f, alpha = 0.f;

  size_t part = (dnum - 1) / BLOCK_SIZE;
  size_t m, e;

  float sum_up = 0.f;
  float sum_dn = 0.f;   
  float six_s, siy_s;

  float r_obs = avg_dist[tid];               //The observed average nearest neighbor distance
  float r_exp = 0.5f / sqrtf(dnum / area); // The expected nearest neighbor distance for a random pattern
  float R_S0 = r_obs / r_exp;                //The nearest neighbor statistic

  float u_R = 0.f;
  if(R_S0 >= R_min) u_R = 0.5f-0.5f * cosf(3.1415926f / R_max * (R_S0 - R_min));
  if(R_S0 >= R_max) u_R = 1.f;

  // Determine the appropriate distance-decay parameter alpha by a triangular membership function
  // Adaptive power parameter: a (alpha)
  if(u_R>= 0.f && u_R<=0.1f)  alpha = a1; 
  if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
  if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
  if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
  if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
  if(u_R>0.9f && u_R<=1.f)  alpha = a5;
  alpha *= 0.5f; // Half of the power

  float six_t = ix[tid];
  float siy_t = iy[tid];
  size_t lid = threadIdx.x;
  for(m = 0; m <= part; m++) {  // Weighted Sum  
    size_t num_threads = min((size_t) BLOCK_SIZE, dnum - BLOCK_SIZE *m);
    if (lid < num_threads) {
      sdx[lid] = dx[lid + BLOCK_SIZE * m];
      sdy[lid] = dy[lid + BLOCK_SIZE * m];
      sdz[lid] = dz[lid + BLOCK_SIZE * m];
    }
    __syncthreads();

    for(e = 0; e < BLOCK_SIZE; e++) {
      six_s = six_t - sdx[e];
      siy_s = siy_t - sdy[e];
      dist = (six_s * six_s + siy_s * siy_s);
      t = 1.f / powf(dist, alpha);  sum_dn += t;  sum_up += t * sdz[e];
    }
  }
  iz[tid] = sum_up / sum_dn;
}

int main(int argc, char *argv[])
{
  // query memory

  // size_t reserved_size = ((size_t) 1024) * 1024 * 1024 * 22;

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  printf("GPU memory: %zu free out of %zu total\n", free_bytes, total_bytes);
  size_t reserved_size = free_bytes * 0.991;

  int* reserved = nullptr;
  cudaMalloc((void**) &reserved, reserved_size);
  cudaMemset(reserved, 1, reserved_size);
  cudaDeviceSynchronize();

  cudaMemGetInfo(&free_bytes, &total_bytes);

  const double factor = 3.0;
  const size_t array_elements = free_bytes * factor / 4 / 7; // 7 arrays
  const size_t array_size_in_bytes = array_elements * 4;
  
  const long long native_difference = free_bytes - array_size_in_bytes * 7;
  printf("GPU memory: %zu free out of %zu total\n", free_bytes, total_bytes);
  printf("Native memory footprint: %zu bytes :: %lld byte difference from available memory\n", array_size_in_bytes * 7, native_difference);

  size_t user_ff_bits_array[] = {32, 31, 28, 24, 20, 17};

  for (int i = 0; i < sizeof(user_ff_bits_array) / 8; i++) {
    const size_t user_ff_bits = user_ff_bits_array[i];
    const size_t ff_array_elements = get_ffarray_size_float(array_elements, user_ff_bits);
    const size_t ff_array_size_in_bytes = ff_array_elements * 4;

    const long long fitfloat_difference = array_size_in_bytes * 7 - ff_array_size_in_bytes * 7;
    const long long fitfloat_difference_available = free_bytes - ff_array_size_in_bytes * 7;
    printf("FF-%zu  memory footprint: %zu bytes :: %lld byte difference from native arrays, %lld from available memory\n", user_ff_bits, ff_array_size_in_bytes * 7, fitfloat_difference, fitfloat_difference_available);    
  }

  const size_t numk = array_elements / 1024; // number of points (unit: 1K)
  const int iterations = 1; // repeat kernel execution

  const size_t dnum = numk * 1024;
  const size_t inum = dnum;
  const size_t dnum_size = dnum * sizeof(float);
  const size_t inum_size = inum * sizeof(float);

  printf("dnum = %zu\n", dnum * 7 * 4);

  // return 0;

  // Area of planar region
  const float width = 2000, height = 2000;
  const float area = width * height;

  std::vector<float> dx(dnum), dy(dnum), dz(dnum);
  std::vector<float> avg_dist(dnum);
  std::vector<float> ix(inum), iy(inum), iz(inum);
  std::vector<float> h_iz(inum);

  srand(123);
  for(size_t i = 0; i < dnum; i++)
  {
    dx[i] = rand()/(float)RAND_MAX * 1000;
    dy[i] = rand()/(float)RAND_MAX * 1000;
    dz[i] = rand()/(float)RAND_MAX * 1000;
  }

  for(size_t i = 0; i < inum; i++)
  {
    ix[i] = rand()/(float)RAND_MAX * 1000;
    iy[i] = rand()/(float)RAND_MAX * 1000;
    iz[i] = 0.f;
  }

  for(size_t i = 0; i < dnum; i++)
  {
    avg_dist[i] = rand()/(float)RAND_MAX * 3;
  }

  float *d_dx, *d_dy, *d_dz;
  float *d_avg_dist;
  float *d_ix, *d_iy, *d_iz;

  cudaMallocManaged((void**)&d_dx, dnum_size); 
  cudaMallocManaged((void**)&d_dy, dnum_size); 
  cudaMallocManaged((void**)&d_dz, dnum_size); 
  cudaMallocManaged((void**)&d_avg_dist, dnum_size); 
  cudaMallocManaged((void**)&d_ix, inum_size); 
  cudaMallocManaged((void**)&d_iy, inum_size); 
  cudaMallocManaged((void**)&d_iz, inum_size); 

  cudaMemcpy(d_dx, dx.data(), dnum_size, cudaMemcpyHostToHost); 
  cudaMemcpy(d_dy, dy.data(), dnum_size, cudaMemcpyHostToHost); 
  cudaMemcpy(d_dz, dz.data(), dnum_size, cudaMemcpyHostToHost); 
  cudaMemcpy(d_avg_dist, avg_dist.data(), dnum_size, cudaMemcpyHostToHost); 
  cudaMemcpy(d_ix, ix.data(), inum_size, cudaMemcpyHostToHost); 
  cudaMemcpy(d_iy, iy.data(), inum_size, cudaMemcpyHostToHost);

  dim3 threadsPerBlock (BLOCK_SIZE);
  dim3 blocksPerGrid ((inum + BLOCK_SIZE - 1) / BLOCK_SIZE);

  cudaDeviceSynchronize();

  // Weighted Interpolate using AIDW

 double rts[iterations];
  for (int i = 0; i < iterations; i++) {
    auto start=std::chrono::steady_clock::now();
    AIDW_Kernel_Tiled<<<blocksPerGrid, threadsPerBlock>>>(
      d_dx, d_dy, d_dz, dnum, d_ix, d_iy, d_iz, inum, area, d_avg_dist);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time * 1e-9f;
  }
  printf("~ aidw: %f (s)\n", rts[iterations / 2]);


  cudaFree(d_dx);
  cudaFree(d_dy);
  cudaFree(d_dz);
  cudaFree(d_ix);
  cudaFree(d_iy);
  cudaFree(d_iz);
  cudaFree(d_avg_dist);
  return 0;
}
