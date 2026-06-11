
GPU_ARCH=sm_89

nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=23 QW-test.cu -o qwtest
./qwtest

nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=19 QW-test.cu -o qwtest
./qwtest

nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=15 QW-test.cu -o qwtest
./qwtest

nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=11 QW-test.cu -o qwtest
./qwtest

