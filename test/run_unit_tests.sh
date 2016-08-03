#!/bin/sh

BUILDDIR="../build/release"
$BUILDDIR/tests/src/test_AurynVector && $BUILDDIR/tests/src/test_EulerTrace && $BUILDDIR/tests/src/test_ComplexMatrix
