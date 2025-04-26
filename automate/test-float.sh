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

ID=$(date +%Y.%m.%d..%H.%M.%S)
RESULTS_DIR="./automate/res_float"

# $# does not include itself in the count of arguments
if [[ $# -ne 1 ]]; then
    echo "USAGE: $0 GPUARCH"
    exit 1
fi

GPUARCH=$1

FILE=FitFloat-Tests.cu

RUNS=9

echo $ID
mkdir -p $RESULTS_DIR

declare -a expo=( 4  5  6  7  8)
declare -a mant=(12 13 14 15 16 17 18 19 20 21 22 23)

for e in "${expo[@]}"
do
    for m in "${mant[@]}"
    do
        printf -v pe "%02d" $e
        printf -v pm "%02d" $m

        outputfile="$RESULTS_DIR/FF..float..$pe$pm.results"

        echo $outputfile

        ./automate/compile_32.sh $e $m $GPUARCH $FILE

        ./flexfloat $RUNS 2 > $outputfile # add 

        echo -e "!!!!!\n" >> $outputfile

        ./flexfloat $RUNS 8 >> $outputfile # count
    done
    echo 
done

for e in "${expo[@]}"
do
    for m in "${mant[@]}"
    do
        printf -v pe "%02d" $e
        printf -v pm "%02d" $m

        outputfile="$RESULTS_DIR/FF..s..float..$pe$pm.results"

        echo $outputfile

        ./automate/compile_32.sh $e $m $GPUARCH $FILE

        ./flexfloat $RUNS 3 > $outputfile # add 

        echo -e "!!!!!\n" >> $outputfile
    done
    echo 
done

outputfileTMP="$RESULTS_DIR/TMP..IEEE..float.results"

./automate/compile_32.sh 8 23 $GPUARCH $FILE
./flexfloat $RUNS 4 > $outputfileTMP # add
echo -e "!!!!!\n" >> $outputfileTMP
./flexfloat $RUNS 9 >> $outputfileTMP # count

for e in "${expo[@]}"
do
    for m in "${mant[@]}"
    do
        printf -v pe "%02d" $e
        printf -v pm "%02d" $m

        outputfile="$RESULTS_DIR/IEEE..float..$pe$pm.results"

        echo $outputfile

        cat $outputfileTMP > $outputfile
    done
    echo 
done

echo "done."
