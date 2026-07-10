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
    FFArr32 dx, 
    FFArr32 dy,
    FFArr32 dz,
    const int dnum,
    FFArr32 ix,
    FFArr32 iy,
    FFArr32 iz,
    const int inum,
    const float area,
    FFArr32 avg_dist) 

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
    FFArr32 dx, 
    FFArr32 dy,
    FFArr32 dz,
    const int dnum,
    FFArr32 ix,
    FFArr32 iy,
    FFArr32 iz,
    const int inum,
    const float area,
    FFArr32 avg_dist) 
{
  // Shared Memory
  __shared__ float sdx[BLOCK_SIZE];
  __shared__ float sdy[BLOCK_SIZE];
  __shared__ float sdz[BLOCK_SIZE];

  int tid = threadIdx.x + blockIdx.x * blockDim.x; 
  if (tid >= inum) return;

  float dist = 0.f, t = 0.f, alpha = 0.f;

  int part = (dnum - 1) / BLOCK_SIZE;
  int m, e;

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
  int lid = threadIdx.x;
  for(m = 0; m <= part; m++) {  // Weighted Sum  
    int num_threads = min(BLOCK_SIZE, dnum - BLOCK_SIZE *m);
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
  const int numk = 512; // number of points (unit: 1K)
  const int iterations = 1; // repeat kernel execution

  const int dnum = numk * 1024;
  const int inum = dnum;
  const size_t dnum_size = dnum * sizeof(float);
  const size_t inum_size = inum * sizeof(float);

  // Area of planar region
  const float width = 2000, height = 2000;
  const float area = width * height;

  std::vector<float> dx(dnum), dy(dnum), dz(dnum);
  std::vector<float> avg_dist(dnum);
  std::vector<float> ix(inum), iy(inum), iz(inum);
  std::vector<float> h_iz(inum);

  srand(123);
  for(int i = 0; i < dnum; i++)
  {
    dx[i] = rand()/(float)RAND_MAX * 1000;
    dy[i] = rand()/(float)RAND_MAX * 1000;
    dz[i] = rand()/(float)RAND_MAX * 1000;
  }

  for(int i = 0; i < inum; i++)
  {
    ix[i] = rand()/(float)RAND_MAX * 1000;
    iy[i] = rand()/(float)RAND_MAX * 1000;
    iz[i] = 0.f;
  }

  for(int i = 0; i < dnum; i++)
  {
    avg_dist[i] = rand()/(float)RAND_MAX * 3;
  }

  FFArr32 d_dx(dnum);
  FFArr32 d_dy(dnum);
  FFArr32 d_dz(dnum);

  FFArr32 d_avg_dist(dnum);

  FFArr32 d_ix(inum);
  FFArr32 d_iy(inum);
  FFArr32 d_iz(inum);

  FloatH2FFDmemcpy(d_dx, dx.data(), dnum);
  FloatH2FFDmemcpy(d_dy, dy.data(), dnum);
  FloatH2FFDmemcpy(d_dz, dz.data(), dnum);
  FloatH2FFDmemcpy(d_avg_dist, avg_dist.data(), dnum);
  FloatH2FFDmemcpy(d_ix, ix.data(), inum);
  FloatH2FFDmemcpy(d_iy, iy.data(), inum);

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


  return 0;
}
