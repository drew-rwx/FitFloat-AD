#include <iostream>
#include <cstdlib>
#include <chrono>
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

#define p_IJWID 6
#define p_JID   4
#define p_JWID  5
#define p_Np    512
#define p_Nq    8
#define p_Nvgeo 12
#define p_RXID  0
#define p_RYID  1
#define p_RZID  7
#define p_SXID  2
#define p_SYID  3
#define p_SZID  8
#define p_TXID  9
#define p_TYID  10
#define p_TZID  11
#define p_cubNp 4096
#define p_cubNq 16

#define dfloat double
#define dlong unsigned long long

// kernel
#include "adv.cu"

dfloat *drandAlloc(dlong N){
  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));
  for(dlong n = 0; n < N; ++n) v[n] = drand48();
  return v;
}

int main(int argc, char **argv) {
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

  const size_t array_elements = free_bytes * factor / 8 / 7; // 7 arrays

  const dlong N = 9;
  const dlong cubN = 9;
  const dlong Nelements = array_elements / 9 / 12;
  const dlong Ntests = 3;
  const dlong Nq = N+1;
  const dlong cubNq = cubN+1;
  const dlong Np = Nq*Nq*Nq;
  const dlong cubNp = cubNq*cubNq*cubNq;
  const dlong offset = Nelements*Np;

  srand48(123);
  dfloat *vgeo           = drandAlloc(Np*Nelements*p_Nvgeo);
  dfloat *cubvgeo        = drandAlloc(cubNp*Nelements*p_Nvgeo);
  dfloat *cubDiffInterpT = drandAlloc(3*cubNp*Nelements);
  dfloat *cubInterpT     = drandAlloc(Np*cubNp);
  dfloat *u              = drandAlloc(3*Np*Nelements);
  dfloat *adv            = drandAlloc(3*Np*Nelements);

  dfloat *d_vgeo, *d_cubvgeo, *d_cubDiffInterpT, *d_cubInterpT, *d_u, *d_adv;
  cudaMallocManaged((void**)&d_vgeo, Np*Nelements*p_Nvgeo*sizeof(dfloat));
  cudaMallocManaged((void**)&d_cubvgeo, cubNp*Nelements*p_Nvgeo*sizeof(dfloat));
  cudaMallocManaged((void**)&d_cubDiffInterpT,3*cubNp*Nelements*sizeof(dfloat));
  cudaMallocManaged((void**)&d_cubInterpT, Np*cubNp*sizeof(dfloat));
  cudaMallocManaged((void**)&d_u, 3*Np*Nelements*sizeof(dfloat));
  cudaMallocManaged((void**)&d_adv, 3*Np*Nelements*sizeof(dfloat));

  cudaMemcpy(d_vgeo, vgeo, Np*Nelements*p_Nvgeo*sizeof(dfloat), cudaMemcpyHostToHost);
  cudaMemcpy(d_cubvgeo, cubvgeo, cubNp*Nelements*p_Nvgeo*sizeof(dfloat), cudaMemcpyHostToHost);
  cudaMemcpy(d_cubDiffInterpT, cubDiffInterpT, 3*cubNp*Nelements*sizeof(dfloat), cudaMemcpyHostToHost);
  cudaMemcpy(d_cubInterpT, cubInterpT, Np*cubNp*sizeof(dfloat), cudaMemcpyHostToHost);
  cudaMemcpy(d_u, u, 3*Np*Nelements*sizeof(dfloat), cudaMemcpyHostToHost);
  cudaMemcpy(d_adv, adv, 3*Np*Nelements*sizeof(dfloat), cudaMemcpyHostToHost);

  cudaDeviceSynchronize();

  // run kernel
  double rts[Ntests];
  for(int test=0;test<Ntests;++test) {
    auto start=std::chrono::steady_clock::now();
    advCubatureHex3D<<<dim3(Nelements, 1), dim3(16, 16)>>>( 
        Nelements,
        d_vgeo,
        d_cubvgeo,
        d_cubDiffInterpT,
        d_cubInterpT,
        offset,
        d_u,
        d_adv);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono:: duration_cast<std::chrono::nanoseconds>(end - start).count();
    rts[test] = time * 1e-3f;
  }
  printf("~ adv: %f (us)\n", rts[Ntests / 2]);


  cudaFree(d_vgeo);
  cudaFree(d_cubvgeo);
  cudaFree(d_cubDiffInterpT);
  cudaFree(d_cubInterpT);
  cudaFree(d_u);
  cudaFree(d_adv);

  free(vgeo          );
  free(cubvgeo       );
  free(cubDiffInterpT);
  free(cubInterpT    );
  free(u             );
  free(adv           );
  return 0;
}
