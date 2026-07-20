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
    BITS_EXPO_64=$1
    BITS_MANT_64=$2
    GPUARCH=$3
    FILE=$4
    EXEC=$5

    echo "Compiling $FILE with $BITS_EXPO_64 exponent bits and $BITS_MANT_64 mantissa bits."

    nvcc -O3 -Xcompiler -fopenmp -w -arch=$GPUARCH -DBITS_FOR_EXPONENT_64=$BITS_EXPO_64 -DBITS_FOR_MANTISSA_64=$BITS_MANT_64 -DUVM1 $INPUT_SIZE_FLAG $FILE -o $EXEC
}

# default params

BITS_EXPO_64=$1
BITS_MANT_64=$2
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

