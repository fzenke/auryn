#!/bin/sh


# Run COBA tests
cd coba_test
./run_coba_test.sh && ./run_coba_parallel_test.sh
