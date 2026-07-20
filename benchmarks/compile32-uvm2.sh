#!/usr/bin/env bash

# This file is part of FitFloat, a drop-in floating-point array replacement supporting user-specified precision on GPUs with the goal of reducing storage requirements.
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Andrew Rodriguez, and Martin Burtscher
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# URL: The latest version of this code is available at https://github.com/burtscher/FitFloat/.
#
# Publication: This work is described in detail in the following paper.
# Andrew Rodriguez, and Martin Burtscher. "FitFloat: Read/Write Random-Access Compressed Floating-Point Arrays for GPUs"
#
# Sponsor: This material is based upon work supported by the U.S. National Science Foundation under Grant Number 2403380 and by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under Award Number DE-SC0022223.

function compile-file () {
    BITS_EXPO_32=$1
    BITS_MANT_32=$2
    GPUARCH=$3
    FILE=$4
    EXEC=$5

    echo "Compiling $FILE with $BITS_EXPO_32 exponent bits and $BITS_MANT_32 mantissa bits."

    nvcc -O3 -Xcompiler -fopenmp -w -arch=$GPUARCH -DBITS_FOR_EXPONENT_32=$BITS_EXPO_32 -DBITS_FOR_MANTISSA_32=$BITS_MANT_32 -DUVM2 $INPUT_SIZE_FLAG $FILE -o $EXEC
}

# default params

BITS_EXPO_32=$1
BITS_MANT_32=$2
GPUARCH=$3

use_limited_configuration_set=$4

if [[ "$use_limited_configuration_set" = false ]]; then
    INPUT_SIZE_FLAG=""
else
    INPUT_SIZE_FLAG="-DINPUT_SIZE_SMALL"
fi

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

