#!/bin/sh

BUILDDIR="../build/release"
$BUILDDIR/test/src/test_AurynVector && \
$BUILDDIR/test/src/test_AurynStateVector && \
$BUILDDIR/test/src/test_EulerTrace && \
$BUILDDIR/test/src/test_LinearTrace && \
$BUILDDIR/test/src/test_ComplexMatrix && exit 0

exit 1
