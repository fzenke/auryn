CC = mpic++

.SECONDARY:

DISTRO:= $(shell source /etc/os-release && echo $$NAME)
$(info $$DISTRO is [${DISTRO}])

ifeq ($(DISTRO),Fedora)
	LDFLAGS=-lboost_program_options -lboost_mpi -lboost_serialization  -L/usr/lib64/mpich/lib/
	CFLAGS=-ansi -pipe -O3 -march=native -ffast-math -pedantic -I/usr/include -I$(SRCDIR) -I/usr/include/mpich-x86_64
else
	CFLAGS=-ansi -pipe -O3 -march=native -ffast-math -pedantic -I/usr/include -I$(SRCDIR)
	LDFLAGS=-lboost_program_options -lboost_mpi -lboost_serialization
endif

include ../Makefile.include
