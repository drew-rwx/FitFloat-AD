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

BITS_EXPO_32=8
BITS_MANT_32=23
BITS_EXPO_64=$1
BITS_MANT_64=$2
GPUARCH=$3

#
#
#


# rodinia-adv (double)

DIR=rodinia-adv


FILE=$DIR/main-ff.cu
EXEC=adv-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# rodinia-asta (double)

DIR=rodinia-asta


FILE=$DIR/main-ff.cu
EXEC=asta-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC



# rodinia-burger (double)

DIR=rodinia-burger


FILE=$DIR/main-ff.cu
EXEC=burger-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


