#!/usr/bin/env bash

function compile-file () {
    BITS_EXPO_32=$1
    BITS_MANT_32=$2
    GPUARCH=$3
    FILE=$4
    EXEC=$5

    echo "Compiling $FILE with $BITS_EXPO_32 exponent bits and $BITS_MANT_32 mantissa bits."

    nvcc -O3 -Xcompiler -fopenmp -arch=$GPUARCH -DBITS_FOR_EXPONENT_32=$BITS_EXPO_32 -DBITS_FOR_MANTISSA_32=$BITS_MANT_32 -DUVM2 $FILE -o $EXEC
}

# default params

BITS_EXPO_32=$1
BITS_MANT_32=$2
GPUARCH=$3

#
#
#


# chi2

DIR=chi2-cuda

FILE=$DIR/chi2-uvm.cu
EXEC=chi2
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/chi2-ff-uvm.cu
EXEC=chi2-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# fhd

DIR=fhd-cuda

FILE=$DIR/main-uvm.cu
EXEC=fhd
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=fhd-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# accuracy

DIR=rodinia-accuracy

FILE=$DIR/main-uvm.cu
EXEC=accuracy
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=accuracy-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# adam

DIR=rodinia-adam

FILE=$DIR/main-uvm.cu
EXEC=adam
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=adam-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# adam-opt

DIR=rodinia-adam

FILE=$DIR/main-uvm-opt.cu
EXEC=adam-opt
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm-opt.cu
EXEC=adam-opt-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# aidw

DIR=rodinia-aidw

FILE=$DIR/main-uvm.cu
EXEC=aidw
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=aidw-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# attention

DIR=rodinia-attention

FILE=$DIR/main-uvm.cu
EXEC=attention
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=attention-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# bilateral

DIR=rodinia-bilateral

FILE=$DIR/main-uvm.cu
EXEC=bilateral
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=bilateral-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# bincount

DIR=rodinia-bincount

FILE=$DIR/main-uvm.cu
EXEC=bincount
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=bincount-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# bscholes

DIR=rodinia-bscholes

FILE=$DIR/blackScholesAnalyticEngine-uvm.cu
EXEC=bscholes
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/blackScholesAnalyticEngine-ff-uvm.cu
EXEC=bscholes-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# bsearch

DIR=rodinia-bsearch

FILE=$DIR/main-uvm.cu
EXEC=bsearch
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=bsearch-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC


# car

DIR=rodinia-car

FILE=$DIR/main-uvm.cu
EXEC=car
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=car-ff
compile-file $BITS_EXPO_32 $BITS_MANT_32 $GPUARCH $FILE $EXEC

