#!/usr/bin/env bash


# $# does not include itself in the count of arguments
if [[ $# -lt 1 ]]; then
    echo "USAGE: $0 GPUARCH"
    exit 1
fi

GPUARCH=$1

RESULTS_DIR="./results/uvm2"

mkdir -p $RESULTS_DIR

declare -a expo=( 8)
declare -a mant=( 8 11 14 17 20 23)

use_limited_configuration_set=$2

if [[ "$use_limited_configuration_set" = false ]]; then
    declare -a expo=( 4  5  6  7  8)
    declare -a mant=(12 13 14 15 16 17 18 19 20 21 22 23)
fi

for e in "${expo[@]}"
do
    for m in "${mant[@]}"
    do
        printf -v pe "%02d" $e
        printf -v pm "%02d" $m

        outputfile="$RESULTS_DIR/FF..float..$pe$pm.results"

        echo $outputfile

        cd benchmarks
        ./compile32-uvm2.sh $e $m $GPUARCH $use_limited_configuration_set 2>/dev/null
        cd ..

        ./benchmarks/accuracy-ff > $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/adam-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/aidw-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/attention-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/bilateral-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/bincount-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/bscholes-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/bsearch-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/car-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/chi2-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/fhd-ff >> $outputfile
        echo -e "!!!!!\n" >> $outputfile
        ./benchmarks/adam-opt-ff >> $outputfile
        echo""
    done
    echo ""
done

outputfileTMP="$RESULTS_DIR/TMP..IEEE..float.results"

cd benchmarks
./compile32-uvm2.sh 8 23 $GPUARCH $use_limited_configuration_set 2>/dev/null
cd ..

./benchmarks/accuracy > $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/adam >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/aidw >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/attention >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/bilateral >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/bincount >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/bscholes >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/bsearch >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/car >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/chi2 >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/fhd >> $outputfileTMP
echo -e "!!!!!\n" >> $outputfileTMP
./benchmarks/adam-opt >> $outputfileTMP
echo""

for e in "${expo[@]}"
do
    for m in "${mant[@]}"
    do
        printf -v pe "%02d" $e
        printf -v pm "%02d" $m

        outputfile="$RESULTS_DIR/IEEE..float..$pe$pm.results"

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
