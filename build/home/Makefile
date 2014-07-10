SRCDIR=../../src
SIMDIR=../../sim
CC = mpic++

.SECONDARY:

CFLAGS=-ansi -Wall -pipe -O3 -march=native -ffast-math -pedantic -I/usr/include -I$(SRCDIR)
LDFLAGS=-lboost_program_options -lboost_mpi -lboost_serialization 

include ../Makefile.include
