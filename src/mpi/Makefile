CXXFLAGS += -Wall -Wextra -g -std=c99 -O3 -DNDEBUG

.phony: all pfs release

all: release

release: pfs-mpi.cpp
	mpic++ pfs-mpi.cpp -o pfs-mpi $(CXXFLAGS)

clean:
	rm -f ./pfs-mpi
	rm -f ./pfs-mpi