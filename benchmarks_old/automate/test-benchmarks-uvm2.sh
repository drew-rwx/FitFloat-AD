#!/usr/bin/env bash

ID=$(date +%Y.%m.%d..%H.%M.%S)
RESULTS_DIR="./automate/results_$ID"

# $# does not include itself in the count of arguments
if [[ $# -ne 1 ]]; then
    echo "USAGE: $0 GPUARCH"
    exit 1
fi

GPUARCH=$1

echo $ID
mkdir -p $RESULTS_DIR

# declare -a expo=(4)
# declare -a mant=(28)
declare -a expo=( 4  5  6  7  8  9 10 11)
declare -a mant=(28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52)


for e in "${expo[@]}"
do
    for m in "${mant[@]}"
    do
        printf -v pe "%02d" $e
        printf -v pm "%02d" $m

        outputfile="$RESULTS_DIR/FF..double..benchmarks..$pe$pm.results"

        echo $outputfile

        cd benchmarks
        ./compile_64-uvm.sh $e $m $GPUARCH 2>/dev/null
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

outputfileTMP="$RESULTS_DIR/TMP..IEEE..double..benchmarks.results"

cd benchmarks
./compile_64-uvm.sh 11 52 $GPUARCH 2>/dev/null
cd ..

./benchmarks/adv-ff > $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/asta-ff >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/burger-ff >> $outputfileTMP
echo""

for e in "${expo[@]}"
do
    for m in "${mant[@]}"
    do
        printf -v pe "%02d" $e
        printf -v pm "%02d" $m

        outputfile="$RESULTS_DIR/IEEE..double..benchmarks..$pe$pm.results"

        echo $outputfile

        cat $outputfileTMP > $outputfile
        echo""
    done
    echo ""
done

echo "done."
