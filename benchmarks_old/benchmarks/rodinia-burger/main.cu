#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "kernels.h"

int main(int argc, char* argv[])
{
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  printf("%.2f GB (%zu Bytes) free, %.2f GB (%zu Bytes) total.\n", 1.0 * free_bytes / 1000000000, free_bytes, 1.0 * total_bytes / 1000000000, total_bytes);
  const size_t num_elements = total_bytes / 8 * 5 / 10 / 32 * 32;

  // Define the domain
  const size_t x_points = sqrt(num_elements) / 2;
  const size_t y_points = x_points;
  const double x_len = 2.0;
  const double y_len = 2.0;
  const double del_x = x_len/(x_points-1);
  const double del_y = y_len/(y_points-1);

  const size_t grid_elems = x_points * y_points;
  const size_t grid_size = sizeof(double) * grid_elems;

  printf("Using %f GB\n", (1.0 * grid_elems*sizeof(double)*4) / 1000000000);

  double *x = (double*) malloc (sizeof(double) * x_points);
  double *y = (double*) malloc (sizeof(double) * y_points);
  double *u = (double*) malloc (grid_size);
  double *v = (double*) malloc (grid_size);
  double *u_new = (double*) malloc (grid_size);
  double *v_new = (double*) malloc (grid_size);

  // store device results
  double *du = (double*) malloc (grid_size);
  double *dv = (double*) malloc (grid_size);

  // Define the parameters
  const int num_itrs = 9;     // Number of time iterations
  const double nu = 0.01;
  const double sigma = 0.0009;
  const double del_t = sigma * del_x * del_y / nu;      // CFL criteria

  printf("2D Burger's equation\n");
  printf("Grid dimension: x = %lld y = %lld\n", x_points, y_points);

  for(size_t i = 0; i < x_points; i++) x[i] = i * del_x;
  for(size_t i = 0; i < y_points; i++) y[i] = i * del_y;

  for(size_t i = 0; i < y_points; i++){
    for(size_t j = 0; j < x_points; j++){
      u[idx(i,j)] = 1.0;
      v[idx(i,j)] = 1.0;
      u_new[idx(i,j)] = 1.0;
      v_new[idx(i,j)] = 1.0;

      if(x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0){
        u[idx(i,j)] = 2.0;
        v[idx(i,j)] = 2.0;
        u_new[idx(i,j)] = 2.0;
        v_new[idx(i,j)] = 2.0;
      }
    }
  }

  double *d_u_new;
  cudaMalloc((void**)&d_u_new, grid_size);

  double *d_v_new;
  cudaMalloc((void**)&d_v_new, grid_size);

  double *d_u;
  cudaMalloc((void**)&d_u, grid_size);

  double *d_v;
  cudaMalloc((void**)&d_v, grid_size);

  cudaMemcpy(d_u_new, u_new, grid_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_new, v_new, grid_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u, u, grid_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, grid_size, cudaMemcpyHostToDevice);

  // ranges of the four kernels
  dim3 grid ((x_points-2+15)/16, (y_points-2+15)/16);
  dim3 block (16, 16);
  dim3 grid2 ((x_points+255)/256);
  dim3 block2 (256);
  dim3 grid3 ((y_points+255)/256);
  dim3 block3 (256);
  dim3 grid4 ((grid_elems+255)/256);
  dim3 block4 (256);

  cudaDeviceSynchronize();
  
  double rts[num_itrs];
  for(int itr = 0; itr < num_itrs; itr++){
    auto start=std::chrono::steady_clock::now();

    core<<<grid, block>>>(d_u_new, d_v_new, d_u, d_v, x_points, y_points, nu, del_t, del_x, del_y);

    // Boundary conditions
    bound_h<<<grid2, block2>>>(d_u_new, d_v_new, x_points, y_points);

    bound_v<<<grid3, block3>>>(d_u_new, d_v_new, x_points, y_points);

    // Updating older values to newer ones
    update<<<grid4, block4>>>(d_u, d_v, d_u_new, d_v_new, grid_elems);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[itr] = time * 1e-9f;
  }
  printf("~ burger: %f (s)\n", rts[num_itrs / 2]);

  free(x);
  free(y);
  free(u);
  free(v);
  free(du);
  free(dv);
  free(u_new);
  free(v_new);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_u_new);
  cudaFree(d_v_new);

  return 0;
}
