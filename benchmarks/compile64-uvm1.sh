#!/usr/bin/env bash

function compile-file () {
    BITS_EXPO_64=$1
    BITS_MANT_64=$2
    GPUARCH=$3
    FILE=$4
    EXEC=$5

    echo "Compiling $FILE with $BITS_EXPO_64 exponent bits and $BITS_MANT_64 mantissa bits."

    nvcc -O3 -Xcompiler -fopenmp -arch=$GPUARCH -DBITS_FOR_EXPONENT_64=$BITS_EXPO_64 -DBITS_FOR_MANTISSA_64=$BITS_MANT_64 -DUVM1 $FILE -o $EXEC
}

# default params

BITS_EXPO_64=$1
BITS_MANT_64=$2
GPUARCH=$3

#
#
#

# adv

DIR=rodinia-adv

FILE=$DIR/main-uvm.cu
EXEC=adv
compile-file $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=adv-ff
compile-file $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# asta

DIR=rodinia-asta

FILE=$DIR/main-uvm.cu
EXEC=asta
compile-file $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=asta-ff
compile-file $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC


# burger

DIR=rodinia-burger

FILE=$DIR/main-uvm.cu
EXEC=burger
compile-file $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

FILE=$DIR/main-ff-uvm.cu
EXEC=burger-ff
compile-file $BITS_EXPO_64 $BITS_MANT_64 $GPUARCH $FILE $EXEC

