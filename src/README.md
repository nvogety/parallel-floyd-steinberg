## Usage
The sequential, OpenMP, and CUDA versions all spawn from `./main.cpp`. You must edit the path to the image and `NCORES` for OpenMP within `./main.cpp` itself.

The MPI version can be found in `./mpi/pfs-mpi.cpp`. You must also update the path to the image within this file, but the number of processors should be specified in the standard `mpirun` command. 