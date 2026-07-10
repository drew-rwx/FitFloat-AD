#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda.h>

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
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  printf("%.2f GB (%zu Bytes) free, %.2f GB (%zu Bytes) total.\n", 1.0 * free_bytes / 1000000000, free_bytes, 1.0 * total_bytes / 1000000000, total_bytes);
  const size_t num_elements = total_bytes / 8 * 1 / 10000 / 5 / 32 * 32;
  printf("data has %zu elements!\n", num_elements);

  const dlong N = 9;
  const dlong cubN = 9;
  const dlong Nelements = num_elements;
  const dlong Ntests = 9;
  const dlong Nq = N+1;
  const dlong cubNq = cubN+1;
  const dlong Np = Nq*Nq*Nq;
  const dlong cubNp = cubNq*cubNq*cubNq;
  const dlong offset = Nelements*Np;

  printf("Data type in bytes: %zu\n", sizeof(dfloat));

  srand48(123);
  dfloat *vgeo           = drandAlloc(Np*Nelements*p_Nvgeo);
  dfloat *cubvgeo        = drandAlloc(cubNp*Nelements*p_Nvgeo);
  dfloat *cubDiffInterpT = drandAlloc(3*cubNp*Nelements);
  dfloat *cubInterpT     = drandAlloc(Np*cubNp);
  dfloat *u              = drandAlloc(3*Np*Nelements);
  dfloat *adv            = drandAlloc(3*Np*Nelements);

  dfloat *d_vgeo, *d_cubvgeo, *d_cubDiffInterpT, *d_cubInterpT, *d_u, *d_adv;
  cudaMalloc((void**)&d_vgeo, Np*Nelements*p_Nvgeo*sizeof(dfloat));
  cudaMalloc((void**)&d_cubvgeo, cubNp*Nelements*p_Nvgeo*sizeof(dfloat));
  cudaMalloc((void**)&d_cubDiffInterpT,3*cubNp*Nelements*sizeof(dfloat));
  cudaMalloc((void**)&d_cubInterpT, Np*cubNp*sizeof(dfloat));
  cudaMalloc((void**)&d_u, 3*Np*Nelements*sizeof(dfloat));
  cudaMalloc((void**)&d_adv, 3*Np*Nelements*sizeof(dfloat));

  cudaMemcpy(d_vgeo, vgeo, Np*Nelements*p_Nvgeo*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cubvgeo, cubvgeo, cubNp*Nelements*p_Nvgeo*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cubDiffInterpT, cubDiffInterpT, 3*cubNp*Nelements*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cubInterpT, cubInterpT, Np*cubNp*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u, u, 3*Np*Nelements*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(d_adv, adv, 3*Np*Nelements*sizeof(dfloat), cudaMemcpyHostToDevice);

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
