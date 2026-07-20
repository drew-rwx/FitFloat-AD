#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cuda.h>

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

#ifndef Real_t 
#define Real_t float
#endif

template <typename T>
__global__ void
kernel_BS (T acc_a,
           T acc_z,
            size_t* __restrict__ acc_r,
           const size_t n)
{ 
  size_t i = blockIdx.x*blockDim.x+threadIdx.x;
  float z = acc_z[i];
  size_t low = 0;
  size_t high = n;
  while (high - low > 1) {
    size_t mid = low + (high - low)/2;
    if (z < acc_a[mid])
      high = mid;
    else
      low = mid;
  }
  acc_r[i] = low;
}

template <typename T>
__global__ void
kernel_BS2 (T acc_a,
            T acc_z,
             size_t* __restrict__ acc_r,
            const size_t n)
{
  size_t i = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned  nbits = 0;
  while (n >> nbits) nbits++;
  size_t k = 1ULL << (nbits - 1);
  float z = acc_z[i];
  size_t idx = (acc_a[k] <= z) ? k : 0;
  while (k >>= 1) {
    size_t r = idx | k;
    if (r < n && z >= acc_a[r]) { 
      idx = r;
    }
  }
  acc_r[i] = idx;
}

template <typename T>
__global__ void
kernel_BS3 (T acc_a,
            T acc_z,
             size_t* __restrict__ acc_r,
            const size_t n)
{
  size_t i = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned nbits = 0;
  while (n >> nbits) nbits++;
  size_t k = 1ULL << (nbits - 1);
  float z = acc_z[i];
  size_t idx = (acc_a[k] <= z) ? k : 0;
  while (k >>= 1) {
    size_t r = idx | k;
    size_t w = r < n ? r : n; 
    if (z >= acc_a[w]) { 
      idx = r;
    }
  }
  acc_r[i] = idx;
}

template <typename T>
__global__ void
kernel_BS4 (T acc_a,
            T acc_z,
             size_t* __restrict__ acc_r,
            const size_t n)
{
  __shared__  size_t k;

  size_t gid = blockIdx.x*blockDim.x+threadIdx.x;
  size_t lid = threadIdx.x; 

  if (lid == 0) {
    unsigned nbits = 0;
    while (n >> nbits) nbits++;
    k = 1ULL << (nbits - 1);
  }
  __syncthreads();

  size_t p = k;
  float z = acc_z[gid];
  size_t idx = (acc_a[p] <= z) ? p : 0;
  while (p >>= 1) {
    size_t r = idx | p;
    size_t w = r < n ? r : n;
    if (z >= acc_a[w]) { 
      idx = r;
    }
  }
  acc_r[gid] = idx;
}

template <typename T>
void bs ( const size_t aSize,
    const size_t zSize,
    T d_a,  // N+1
    T d_z,  // T
    size_t *d_r,   // T
    const size_t n,
    const int repeat )
{
  double rts[repeat];
  for (int i = 0; i < repeat; i++) {
    auto start=std::chrono::steady_clock::now();
    kernel_BS<<<zSize/256, 256>>>(d_a, d_z, d_r, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time * 1e-9f;
  }
  printf("~ (bs1) %f (s)\n", rts[repeat / 2]);
}

template <typename T>
void bs2 ( const size_t aSize,
    const size_t zSize,
    T d_a,  // N+1
    T d_z,  // T
    size_t *d_r,   // T
    const size_t n,
    const int repeat )
{
  double rts[repeat];
  for (int i = 0; i < repeat; i++) {
    auto start=std::chrono::steady_clock::now();
    kernel_BS2<<<zSize/256, 256>>>(d_a, d_z, d_r, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time * 1e-9f;
  }
  printf("~ (bs2) %f (s)\n", rts[repeat / 2]);
}

template <typename T>
void bs3 ( const size_t aSize,
    const size_t zSize,
    T d_a,  // N+1
    T d_z,  // T
    size_t *d_r,   // T
    const size_t n,
    const int repeat )
{
  double rts[repeat];
  for (int i = 0; i < repeat; i++) {
    auto start=std::chrono::steady_clock::now();
    kernel_BS3<<<zSize/256, 256>>>(d_a, d_z, d_r, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time * 1e-9f;
  }
  printf("~ (bs3) %f (s)\n", rts[repeat / 2]);
}

template <typename T>
void bs4 ( const size_t aSize,
    const size_t zSize,
    T d_a,  // N+1
    T d_z,  // T
    size_t *d_r,   // T
    const size_t n,
    const int repeat )
{
  double rts[repeat];
  for (int i = 0; i < repeat; i++) {
    auto start=std::chrono::steady_clock::now();
    kernel_BS4<<<zSize/256, 256>>>(d_a, d_z, d_r, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[i] = time * 1e-9f;
  }
  printf("~ (bs4) %f (s)\n", rts[repeat / 2]);
}

#ifdef DEBUG
void verify(Real_t *a, Real_t *z, size_t *r, size_t aSize, size_t zSize, std::string msg)
{
  for (size_t i = 0; i < zSize; ++i)
  {
    // check result
    if (!(r[i]+1 < aSize && a[r[i]] <= z[i] && z[i] < a[r[i] + 1]))
    {
      std::cout << msg << ": incorrect result:" << std::endl;
      std::cout << "index = " << i << " r[index] = " << r[i] << std::endl;
      std::cout << a[r[i]] << " <= " << z[i] << " < " << a[r[i] + 1] << std::endl;
      break;
    }
    // clear result
    r[i] = 0xFFFFFFFF;
  }
}
#endif

int main(int argc, char* argv[])
{
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

  const size_t array_elements = free_bytes * factor / 4 / 7; // 7 sections

  size_t numElem = array_elements;
  uint repeat = 3;

  srand(2);
  size_t aSize = numElem;
  size_t zSize = 2*aSize;
  Real_t *a = NULL;
  Real_t *z = NULL;
  size_t *r = NULL;
  posix_memalign((void**)&a, 1024, aSize * sizeof(Real_t));
  posix_memalign((void**)&z, 1024, zSize * sizeof(Real_t));
  posix_memalign((void**)&r, 1024, zSize * sizeof(size_t));

  size_t N = aSize-1;

  // strictly ascending
  for (size_t i = 0; i < aSize; i++) a[i] = i;

  // lower = 0, upper = n-1
  for (size_t i = 0; i < zSize; i++) z[i] = rand() % N;

  size_t *d_r;
  cudaMallocManaged((void**)&d_r, sizeof(size_t)*zSize);

  FFArr32 d_a(aSize);
  FFArr32 d_z(zSize);
  FloatH2FFDmemcpy(d_a, a, aSize);
  FloatH2FFDmemcpy(d_z, z, zSize);

  bs(aSize, zSize, d_a, d_z, d_r, N, repeat);


  bs2(aSize, zSize, d_a, d_z, d_r, N, repeat);



  bs3(aSize, zSize, d_a, d_z, d_r, N, repeat);


  bs4(aSize, zSize, d_a, d_z, d_r, N, repeat);



  cudaFree(d_r);
  free(a);
  free(z);
  free(r);
  return 0;
}
