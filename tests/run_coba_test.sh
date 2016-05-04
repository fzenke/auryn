#!/bin/bash

BUILDDIR="../build/release"

# Compile code
$BUILDDIR/bootstrap.sh

# Benchmark parameters
SIMTIME=10
REVISION=`git log --pretty=oneline -1 | cut -d " " -f 1`

# Vogels-Abbott benchmark, single core
TMPDIR=`mktemp -d`
$BUILDDIR/examples/sim_coba_benchmark --simtime $SIMTIME --dir $TMPDIR
$CMD_BENCHMARK1
md5sum $TMPDIR/*.ras | cut -d " " -f 1 > coba_checksums.txt
rm -r $TMPDIR


# Writ result to file
diff coba_checksums.txt coba_checksums.ref
RETURNVALUE=$?

echo "$REVISION coba-test $RETURNVALUE" >> test_results.dat

if [ $? -ne 0 ]
then echo "The checksums for the coba-test are different!"
fi

exit $RETURNVALUE
