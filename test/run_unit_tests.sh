#!/bin/sh

BUILDDIR="../build/release"

echo "Running unit tests..."
$BUILDDIR/test/src/test_AurynVector && \
$BUILDDIR/test/src/test_AurynStateVector && \
$BUILDDIR/test/src/test_AurynDelayVector && \
$BUILDDIR/test/src/test_EulerTrace && \
$BUILDDIR/test/src/test_LinearTrace && \
$BUILDDIR/test/src/test_ComplexMatrix && exit 0

exit 1
