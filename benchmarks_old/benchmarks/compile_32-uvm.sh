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

# adam

DIR=rodinia-adam

FILE=$DIR/main-uvm.cu
EXEC=adam
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=adam-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# langevin

DIR=langevin-cuda

FILE=$DIR/main-uvm.cu
EXEC=langevin
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=langevin-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# fhd

DIR=fhd-cuda

FILE=$DIR/main-uvm.cu
EXEC=fhd
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=fhd-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# chi2

DIR=chi2-cuda

FILE=$DIR/chi2-uvm.cu
EXEC=chi2
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/chi2-ff-uvm.cu
EXEC=chi2-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

# accuracy

DIR=rodinia-accuracy

FILE=$DIR/main-uvm.cu
EXEC=accuracy
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=accuracy-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# aidw

DIR=rodinia-aidw

FILE=$DIR/main-uvm.cu
EXEC=aidw
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=aidw-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# attention

DIR=rodinia-attention

FILE=$DIR/main-uvm.cu
EXEC=attention
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=attention-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# bilateral

DIR=rodinia-bilateral

FILE=$DIR/main-uvm.cu
EXEC=bilateral
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=bilateral-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# bincount

DIR=rodinia-bincount

FILE=$DIR/main-uvm.cu
EXEC=bincount
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=bincount-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# rodinia-bscholes

DIR=rodinia-bscholes

FILE=$DIR/blackScholesAnalyticEngine-uvm.cu
EXEC=bscholes
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/blackScholesAnalyticEngine-ff-uvm.cu
EXEC=bscholes-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# bsearch

DIR=rodinia-bsearch

FILE=$DIR/main-uvm.cu
EXEC=bsearch
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=bsearch-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# car

DIR=rodinia-car

FILE=$DIR/main-uvm.cu
EXEC=car
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=car-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

