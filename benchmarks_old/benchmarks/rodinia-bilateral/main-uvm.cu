#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <chrono>
#include "reference.h"
#include "../../FlexFloat.h"

template<int R>
__global__ void bilateralFilter(
    const float *__restrict__ in,
    float *__restrict__ out,
    int w, 
    int h, 
    float a_square,
    float variance_I,
    float variance_spatial)
{
  const int idx = blockIdx.x*blockDim.x + threadIdx.x;
  const int idy = blockIdx.y*blockDim.y + threadIdx.y;

  if(idx >= w || idy >= h) return;

  int id = idy*w + idx;
  float I = in[id];
  float res = 0;
  float normalization = 0;

  // window centered at the coordinate (idx, idy)
#ifdef LOOP_UNROLL
  #pragma unroll
#endif
  for(int i = -R; i <= R; i++) {
#ifdef LOOP_UNROLL
    #pragma unroll
#endif
    for(int j = -R; j <= R; j++) {

      int idk = idx+i;
      int idl = idy+j;

      // mirror edges
      if( idk < 0) idk = -idk;
      if( idl < 0) idl = -idl;
      if( idk > w - 1) idk = w - 1 - i;
      if( idl > h - 1) idl = h - 1 - j;

      int id_w = idl*w + idk;
      float I_w = in[id_w];

      // range kernel for smoothing differences in intensities
      float range = -(I-I_w) * (I-I_w) / (2.f * variance_I);

      // spatial (or domain) kernel for smoothing differences in coordinates
      float spatial = -((idk-idx)*(idk-idx) + (idl-idy)*(idl-idy)) /
                      (2.f * variance_spatial);

      // the weight is assigned using the spatial closeness (using the spatial kernel) 
      // and the intensity difference (using the range kernel)
      float weight = a_square * expf(spatial + range);

      normalization += weight;
      res += (I_w * weight);
    }
  }
  out[id] = res/normalization;
}

//
// reference https://en.wikipedia.org/wiki/Bilateral_filter
//
int main(int argc, char *argv[]) {
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
  const size_t array_elements = free_bytes * factor / 4;
  const size_t array_size_in_bytes = array_elements * 4;
  
  const long long native_difference = free_bytes - array_size_in_bytes;
  printf("GPU memory: %zu free out of %zu total\n", free_bytes, total_bytes);
  printf("Native memory footprint: %zu bytes :: %lld byte difference from available memory\n", array_size_in_bytes, native_difference);

  size_t user_ff_bits_array[] = {32, 31, 28, 24, 20, 17};

  for (int i = 0; i < sizeof(user_ff_bits_array) / 8; i++) {
    const size_t user_ff_bits = user_ff_bits_array[i];
    const size_t ff_array_elements = get_ffarray_size_float(array_elements, user_ff_bits);
    const size_t ff_array_size_in_bytes = ff_array_elements * 4;

    const long long fitfloat_difference = array_size_in_bytes - ff_array_size_in_bytes;
    const long long fitfloat_difference_available = free_bytes - ff_array_size_in_bytes;
    printf("FF-%zu  memory footprint: %zu bytes :: %lld byte difference from native arrays, %lld from available memory\n", user_ff_bits, ff_array_size_in_bytes, fitfloat_difference, fitfloat_difference_available);    
  }


  // image dimensions
  size_t w = sqrt(array_elements / 2);
  size_t h = w;
  const size_t img_size = w*h;

  printf("Using %f GB\n", (1.0 * img_size*sizeof(float)*2) / 1000000000);


   // As the range parameter increases, the bilateral filter gradually 
   // approaches Gaussian convolution more closely because the range 
   // Gaussian widens and flattens, which means that it becomes nearly
   // constant over the intensity interval of the image.
  float variance_I = 0.5;

   // As the spatial parameter increases, the larger features get smoothened.
  float variance_spatial = 0.5;

  // square of the height of the curve peak
  float a_square = 0.5f / (variance_I * (float)M_PI);

  int repeat = 9;

  float *d_src, *d_dst;
  cudaMallocManaged((void**)&d_dst, img_size * sizeof(float));
  cudaMallocManaged((void**)&d_src, img_size * sizeof(float));

  float *h_src = (float*) malloc (img_size * sizeof(float));
  // host and device results
  float *h_dst = (float*) malloc (img_size * sizeof(float));
  float *r_dst = (float*) malloc (img_size * sizeof(float));

  srand(123);
  for (int i = 0; i < img_size; i++)
    h_src[i] = rand() % 256;

  cudaMemcpy(d_src, h_src, img_size * sizeof(float), cudaMemcpyHostToHost); 

  dim3 threads (16, 16);
  dim3 blocks ((w+15)/16, (h+15)/16);

  cudaDeviceSynchronize();

  double rts[repeat];
  for (int i = 0; i < repeat; i++){
    auto start=std::chrono::steady_clock::now();
    bilateralFilter<3><<<blocks, threads>>>(
        d_src, d_dst, w, h, a_square, variance_I, variance_spatial);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time * 1e-9f;
  }
  printf("~ bilateral (3x3): %f (s)\n", rts[repeat / 2]);

  for (int i = 0; i < repeat; i++){
    auto start=std::chrono::steady_clock::now();
    bilateralFilter<6><<<blocks, threads>>>(
        d_src, d_dst, w, h, a_square, variance_I, variance_spatial);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time * 1e-9f;
  }
  printf("~ bilateral (6x6): %f (s)\n", rts[repeat / 2]);

  for (int i = 0; i < repeat; i++){
    auto start=std::chrono::steady_clock::now();
    bilateralFilter<9><<<blocks, threads>>>(
        d_src, d_dst, w, h, a_square, variance_I, variance_spatial);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time * 1e-9f;
  }
  printf("~ bilateral (9x9): %f (s)\n", rts[repeat / 2]);


  free(h_dst);
  free(r_dst);
  free(h_src);
  cudaFree(d_dst);
  cudaFree(d_src);
  return 0;
}
