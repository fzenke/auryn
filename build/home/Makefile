SRCDIR=../../src
SIMDIR=../../sim
CC = mpic++

.SECONDARY:

CFLAGS=-ansi -Wall -pipe -O3 -march=native -ffast-math -pedantic -I/usr/include -I/usr/include/gsl $(shell pkg-config --cflags gsl) -I$(SRCDIR)
LDFLAGS=$(shell pkg-config --libs gsl) -lboost_program_options -lboost_mpi -lboost_serialization -lboost_filesystem -lboost_system

include ../Makefile.include
