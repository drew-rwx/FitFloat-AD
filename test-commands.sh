
GPU_ARCH=sm_86

nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=23 QW-test.cu -o qwtest
./qwtest

nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=19 QW-test.cu -o qwtest
./qwtest

nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=15 QW-test.cu -o qwtest
./qwtest

nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=11 QW-test.cu -o qwtest
./qwtest

# CODE_DIR=rodinia-attention
# NATIVE_SRC=main-uvm.cu
# FITFLT_SRC=main-ff-uvm.cu

# cd "benchmarks"
# nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=23 -DUVM1 "$CODE_DIR/$NATIVE_SRC" -o native
# nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true ./native

# nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=23 -DUVM1 "$CODE_DIR/$FITFLT_SRC" -o fitflt
# nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true ./fitflt

# nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=19 -DUVM1 "$CODE_DIR/$FITFLT_SRC" -o fitflt
# nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true ./fitflt

# nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=15 -DUVM1 "$CODE_DIR/$FITFLT_SRC" -o fitflt
# nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true ./fitflt

# nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=8 -DBITS_FOR_MANTISSA_32=11 -DUVM1 "$CODE_DIR/$FITFLT_SRC" -o fitflt
# nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true ./fitflt

# nvcc -O3 -Xcompiler -fopenmp -arch=$GPU_ARCH -DBITS_FOR_EXPONENT_32=4 -DBITS_FOR_MANTISSA_32=12 -DUVM1 "$CODE_DIR/$FITFLT_SRC" -o fitflt
# nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --show-output=true ./fitflt

# rm -f report*
# rm fitflt
# rm native
# cd ..

