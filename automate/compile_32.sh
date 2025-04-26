#
# This file is part of FitFloat, a floating-point array representation for GPUs that allows the user to choose the number of bits in the exponent and mantissa fields.
#
# BSD 3-Clause License
#
# Copyright (c) 2025, Andrew Rodriguez and Martin Burtscher
# All rights reserved.
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
# URL: The latest version of this code is available at https://github.com/burtscher/FitFloat.
#
# Sponsor: This code is based upon work supported by the U.S. Department of Energy, National Nuclear Security Administration, under Award Number DE-NA0003969.
#

# $# does not include itself in the count of arguments
if [[ $# -ne 4 ]]; then
    echo "USAGE: $0 BITS_EXPO BITS_MANT GPUARCH FILE"
    exit 1
fi

BITS_EXPO=$1
BITS_MANT=$2
GPUARCH=$3
FILE=$4

echo "Compiling with $BITS_EXPO expo. bits, $BITS_MANT mant. bits..."

if [[ $GPUARCH == "amd" ]]; then
    hipcc -O3 -fopenmp -Wno-unused-result -std=c++17 -DBITS_FOR_EXPONENT_32=$BITS_EXPO -DBITS_FOR_MANTISSA_32=$BITS_MANT $FILE -o flexfloat
else
    nvcc -O3 -Xcompiler -fopenmp -arch=$GPUARCH -DBITS_FOR_EXPONENT_32=$BITS_EXPO -DBITS_FOR_MANTISSA_32=$BITS_MANT $FILE -o flexfloat
fi
