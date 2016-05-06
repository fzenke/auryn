#!/bin/bash

BUILDDIR="../../build/debug"
TOOLDIR="$BUILDDIR/tools"

# Compile code
CURDIR=`pwd`
cd $BUILDDIR  && ./bootstrap.sh 
cd $CURDIR
make -C $BUILDDIR 

# Benchmark parameters
SIMTIME=10
REVISION=`git log --pretty=oneline -1 | cut -d " " -f 1`
TESTNAME=`basename "$0"`

# Vogels-Abbott benchmark, single core
TMPDIR=`mktemp -d`
WMATLOADSTRING="--fee pynn.ee.wmat --fei pynn.ei.wmat --fie pynn.ie.wmat --fii pynn.ii.wmat"
# because the order of spikes on file in multicore is not the same as in single core we only 
# spikes from one rank into our comparison
$BUILDDIR/examples/sim_coba_binmon $WMATLOADSTRING --simtime $SIMTIME --dir $TMPDIR 
echo "Computing checksum ..."
$TOOLDIR/aube -i $TMPDIR/coba.*.e.spk | awk '{ if ($2%2==0) print }' | md5sum > coba_checksums.txt
rm -r $TMPDIR


# Writ result to file
echo "Comparing checksum to reference ..."
diff coba_checksums.txt coba_checksums.ref
RETURNVALUE=$?

echo "$REVISION $TESTNAME $RETURNVALUE" >> test_results.dat

if [ $? -ne 0 ]
then echo "The checksums for the $TESTNAME are different!"
fi

exit $RETURNVALUE
