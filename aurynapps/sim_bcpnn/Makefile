# Update the following lines to your system requirements
AURYNPATH=$(HOME)/anderslan/auryn

# Path to Auryn include files
AURYNINC=$(AURYNPATH)/src

# Path to Auryn library
AURYNLIB=$(AURYNPATH)/build/release/src

# The following should not require updating in most cases 
CXX = mpicxx
CXXFLAGS=-ansi -pipe -O3 -march=native -ffast-math -pedantic -I/usr/include -I$(AURYNINC)
LDFLAGS=$(AURYNLIB)/libauryn.a -lboost_filesystem -lboost_system -lboost_program_options -lboost_mpi -lboost_serialization

# Add your simulation's file name here as default target
all: sim_bcpnn_1

sim_bcpnn_1: sim_bcpnn_1.o
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $(subst .o,,$<)

# sim_bcpnn_1.o : sim_bcpnn_1.cpp
# 	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm *.o sim_bcpnn_1

clean2:
	rm -f *.txt *.log *~

