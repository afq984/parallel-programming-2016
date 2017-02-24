CC                      := gcc
CXX                     := g++
CCFLAGS                 := -O3 -march=native -Wall -std=gnu11
CXXFLAGS                := -O3 -march=native -Wall -std=c++11

NVCC                    := nvcc
NVFLAGS                 := -arch=sm_20 -O3 -use_fast_math -Wno-deprecated-gpu-targets

OMPFLAGS                := -fopenmp
MPILIBS                 := -I/opt/mpich-3.2-gcc/include -L/opt/mpich-3.2-gcc/lib -lmpich
# MPILIBS := -L/usr/lib/openmpi -lmpi

EXES                    := HW4_cuda.exe HW4_openmp.exe HW4_mpi.exe

all: $(EXES)

clean:
	rm -f $(EXES)

HW4_cuda.exe: cuda.cu
	$(NVCC) $(NVFLAGS) -std=c++11 -o $@ $?

HW4_openmp.exe: openmp.cu
	$(NVCC) $(NVFLAGS) -std=c++11 -o $@ $?

HW4_mpi.exe: mpi.cu
	$(NVCC) $(NVFLAGS) $(MPILIBS) -std=c++11 -o $@ $?
