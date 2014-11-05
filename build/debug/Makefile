.SECONDARY:

CC = mpicxx

CFLAGS=-ansi -Wall -pipe -g -ffast-math -march=native -pedantic -I/usr/include -I$(SRCDIR) -I$(DEVDIR)
# LDFLAGS=-L/usr/local/atlas/lib -lgsl -lcblas -latlas -lm -lboost_program_options -lboost_mpi -lboost_filesystem -lboost_system
#
#
LDFLAGS=-lboost_program_options -lboost_mpi -lboost_serialization -lboost_date_time

include ../Makefile.include
