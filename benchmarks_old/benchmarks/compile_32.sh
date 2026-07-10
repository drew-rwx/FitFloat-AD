#!/usr/bin/env bash

function compile-file () {
    BITS_EXPO_32=$1
    BITS_MANT_32=$2
    BITS_EXPO_64=$3
    BITS_MANT_64=$4
    GPUARCH=$5
    FILE=$6
    EXEC=$7

    echo "Compiling $FILE with (float) $BITS_EXPO_32 expo. bits, $BITS_MANT_32 mant. bits, and (double) $BITS_EXPO_64 expo. bits, $BITS_MANT_64 mant. bits..."

    if [[ $GPUARCH == "amd" ]]; then
        hipcc -O3 -fopenmp -Wno-unused-result -std=c++17 -DBITS_FOR_EXPONENT_32=$BITS_EXPO_32 -DBITS_FOR_EXPONENT_64=$BITS_EXPO_64 -DBITS_FOR_MANTISSA_32=$BITS_MANT_32 -DBITS_FOR_MANTISSA_64=$BITS_MANT_64 $FILE -o $EXEC
    else
        nvcc -O3 -Xcompiler -fopenmp -arch=$GPUARCH -DBITS_FOR_EXPONENT_32=$BITS_EXPO_32 -DBITS_FOR_EXPONENT_64=$BITS_EXPO_64 -DBITS_FOR_MANTISSA_32=$BITS_MANT_32 -DBITS_FOR_MANTISSA_64=$BITS_MANT_64 $FILE -o $EXEC
    fi
}

# default params

BITS_EXPO_32=$1
BITS_MANT_32=$2
BITS_EXPO_64=11
BITS_MANT_64=52
GPUARCH=$3

#
#
#

# rodinia-accuracy

DIR=rodinia-accuracy

FILE=$DIR/main.cu
EXEC=accuracy
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC



# rodinia-adam

DIR=rodinia-adam

FILE=$DIR/main.cu
EXEC=adam
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

# rodinia-aidw

DIR=rodinia-aidw

FILE=$DIR/main.cu
EXEC=aidw
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC



# rodinia-attention

DIR=rodinia-attention

FILE=$DIR/main.cu
EXEC=attention
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC



# rodinia-bilateral

DIR=rodinia-bilateral

FILE=$DIR/main.cu
EXEC=bilateral
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC



# rodinia-bincount

DIR=rodinia-bincount

FILE=$DIR/main.cu
EXEC=bincount
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC



# rodinia-bscholes

DIR=rodinia-bscholes

FILE=$DIR/blackScholesAnalyticEngine.cu
EXEC=bscholes
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC



# rodinia-bsearch

DIR=rodinia-bsearch

FILE=$DIR/main.cu
EXEC=bsearch
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC




# rodinia-car

DIR=rodinia-car

FILE=$DIR/main.cu
EXEC=car
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC



