#!/bin/sh

./sim_isp_orig --out /tmp/out --simtime 10
./sim_background --dir /tmp/ --tau 10 --simtime 10
