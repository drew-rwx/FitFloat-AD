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


# $# does not include itself in the count of arguments
if [[ $# -lt 1 ]]; then
    echo "USAGE: $0 GPUARCH"
    exit 1
fi

GPUARCH=$1

RESULTS_DIR="./results/uvm1"

mkdir -p $RESULTS_DIR

declare -a expo=(11)
declare -a mant=(21 24 28 32 36 40 44 48 52)

use_limited_configuration_set=$2

if [[ "$use_limited_configuration_set" = false ]]; then
    declare -a expo=( 4  5  6  7  8  9 10 11)
    declare -a mant=(28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52)
fi

for e in "${expo[@]}"
do
    for m in "${mant[@]}"
    do
        printf -v pe "%02d" $e
        printf -v pm "%02d" $m

        outputfile="$RESULTS_DIR/FF..double..$pe$pm.results"

        echo $outputfile

        cd benchmarks
        ./compile64-uvm1.sh $e $m $GPUARCH $use_limited_configuration_set 2>/dev/null
        cd ..

        ./benchmarks/adv-ff > $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/asta-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/burger-ff >> $outputfile
        echo""
    done
    echo ""
done

outputfileTMP="$RESULTS_DIR/TMP..IEEE..double.results"

cd benchmarks
./compile64-uvm1.sh 11 52 $GPUARCH $use_limited_configuration_set 2>/dev/null
cd ..

./benchmarks/adv > $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/asta >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/burger >> $outputfileTMP
echo""

for e in "${expo[@]}"
do
    for m in "${mant[@]}"
    do
        printf -v pe "%02d" $e
        printf -v pm "%02d" $m

        outputfile="$RESULTS_DIR/IEEE..double..$pe$pm.results"

        echo $outputfile

        cat $outputfileTMP > $outputfile
        echo""
    done
    echo ""
done

rm $outputfileTMP

cd benchmarks
bash clean.sh
cd ..

echo "done."
