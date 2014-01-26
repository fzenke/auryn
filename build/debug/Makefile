SRCDIR=../../src
SIMDIR=../../sim
CC = mpic++

.SECONDARY:

CFLAGS=-ansi -Wall -pipe -g -pedantic -I/usr/include/gsl $(shell pkg-config --cflags gsl) -I$(SRCDIR)
# LDFLAGS=-L/usr/local/atlas/lib -lgsl -lcblas -latlas -lm -lboost_program_options -lboost_mpi -lboost_filesystem -lboost_system
LDFLAGS=$(shell pkg-config --libs gsl) -lboost_program_options -lboost_mpi -lboost_serialization -lboost_filesystem -lboost_system

include ../Makefile.include
