#!/bin/bash

# Compile code
make clean
./bootstrap.sh
sleep 2

# Benchmark parameters
SIMTIME=100
STATRUNS="1 2 3 4 5"
HOSTNAME=`hostname`
DATE=`date +"%Y-%m-%d"`
REVISION=`git log --pretty=oneline -1 | cut -d " " -f 1`
TMPDIR=`mktemp -d`

# Function declaration
function fun_benchmark() 
{
	echo $1
	> $TMPDIR/times.dat
	for VARIABLE in $STATRUNS; do
		$1
		cat $TMPDIR/elapsed.dat >> $TMPDIR/times.dat
	done

	awk '{ for (i=1;i<=NF;i++) { sum[i] += 1.0*$i; sum2[i] += 1.0*$i*$i; } } \
		END { for (i=1;i<=NF;i++) { mean = 1.*sum[i]/NR; var  = 1.*sum2[i]/NR-mean*mean; std = sqrt(var + 1.0/12); \
		   printf "%f  %f\t",mean,std } printf "\n" }' $TMPDIR/times.dat > $TMPDIR/result.dat

	FUNCTION_RESULT=`cat $TMPDIR/result.dat`
	cp $TMPDIR/result.dat last_benchmark_result.dat
}


# Vogels-Abbott benchmark, single core
CMD_BENCHMARK1="examples/sim_coba_benchmark --fast --simtime $SIMTIME --dir $TMPDIR"
fun_benchmark "$CMD_BENCHMARK1"
RESULT_BENCHMARK1=$FUNCTION_RESULT

# Zenke plasticity benchmark, single core
CMD_BENCHMARK2="examples/sim_background --fast --tau 10 --simtime $SIMTIME --dir $TMPDIR"
fun_benchmark "$CMD_BENCHMARK2"
RESULT_BENCHMARK2=$FUNCTION_RESULT

# Zenke plasticity benchmark, two cores
CMD_BENCHMARK3="mpirun -n 2 examples/sim_background --fast --tau 10 --simtime $SIMTIME --dir $TMPDIR"
fun_benchmark "$CMD_BENCHMARK3"
RESULT_BENCHMARK3=$FUNCTION_RESULT


# Writ result to file
echo "$HOSTNAME $REVISION $RESULT_BENCHMARK1 $RESULT_BENCHMARK2 $RESULT_BENCHMARK3 $DATE" >> benchmark_results.dat

# Clean up
rm -r $TMPDIR
