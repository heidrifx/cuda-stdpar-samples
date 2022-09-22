#!/bin/bash
man="\
Usage: aggregateData_local memory_size repreats [clear data]\n \
   memory_size\t must be the number of gigabytes available on the GPU\n \
   repeats\t describes the number of times to repeat each sample\n \
   clear_data\t set to 1 if the data directory should be cleared before\n\
"

# -h and error messages
if [[ "$1" == "-h" || "$1" == "--help" ]]; then printf "$man"; exit; fi
if [[ $# < 2 ]]; then printf "$man"; exit 21; fi

# create data dir if possible
[[ ! -d data/ ]] && mkdir data/

# rename vars
memory=$1
repeats=$2

# clear data if $3 is 1
[ "$3" == 1 ] && rm -f data/*

for file in bin/*
do
    # get file name
    name=$(basename $file)
    # skip multi files
    if [[ "$name" =~ multi ]]; then continue; fi

    csv=data/$name.csv
    regx=(^$name \n)
    echo "Running $name and saving to $csv"
    # create csv if non-existent
    [[ ! -f $csv ]] && touch $csv
    # add header if not already present
    [[ ! $(cat $csv) =~ $regx ]] && gawk -i inplace -v var="$name" 'BEGINFILE{print var} 1' $csv
    # repeat $2/$repeats times
    for i in $(seq 1 $repeats)
    do
        # use $1/$memory GB of memory
        eval "$file $1" | grep -oE "([[:digit:]]+)(µ|m)s" | sed -E "s/(µ|m)s//" >> $csv
    done
done
