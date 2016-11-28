CXX := g++
MPICXX := mpicxx
MBSFLAGS := -DMBS_FIXED_POINT=1 -DMBS_BLACK_HOLE=1
CXXFLAGS := -O3 -march=native -Wall -std=c++11 $(MBS_FLAGS)
LDFLAGS := -lX11
COMMONFLAGS := $(MBSFLAGS) $(CXXFLAGS) $(LDFLAGS)
EVERYTHING := MS_SEQ MS_MPI_static MS_MPI_dynamic MS_OpenMP_static MS_OpenMP_dynamic MS_Hybrid_static MS_Hybrid_dynamic
BENCHMARK := MS_SEQ_base MS_SEQ_fixed MS_SEQ_math
AUTOSOURCECODE := MS_SEQ.cc MS_MPI_static.cc MS_MPI_dynamic.cc MS_OpenMP_static.cc MS_OpenMP_dynamic.cc MS_Hybrid_static.cc MS_Hybrid_dynamic.cc

.PHONY: all
all: $(EVERYTHING)
.PHONY: benchmark
benchmark: $(EVERYTHING) $(BENCHMARK)
.PHONY: clean
clean:
	rm -f $(wildcard $(EVERYTHING) $(BENCHMARK) $(AUTOSOURCECODE))
MS_SEQ: mandelbrot_set.cc Makefile
	$(CXX) $(COMMONFLAGS) \
		-DMBS_DYNAMIC=0 -DMBS_USE_OMP=0 -DMBS_USE_MPI=0 \
		$< -o $@
MS_OpenMP_static: mandelbrot_set.cc Makefile
	$(CXX) $(COMMONFLAGS) -fopenmp \
		-DMBS_DYNAMIC=0 -DMBS_USE_OMP=1 -DMBS_USE_MPI=0 \
		$< -o $@
MS_OpenMP_dynamic: mandelbrot_set.cc Makefile
	$(CXX) $(COMMONFLAGS) -fopenmp \
		-DMBS_DYNAMIC=1 -DMBS_USE_OMP=1 -DMBS_USE_MPI=0 \
		$< -o $@
MS_MPI_static: mandelbrot_set.cc Makefile
	$(MPICXX) $(COMMONFLAGS) \
		-DMBS_DYNAMIC=0 -DMBS_USE_OMP=0 -DMBS_USE_MPI=1 \
		$< -o $@
MS_MPI_dynamic: mandelbrot_set.cc Makefile
	$(MPICXX) $(COMMONFLAGS) \
		-DMBS_DYNAMIC=1 -DMBS_USE_OMP=0 -DMBS_USE_MPI=1 \
		$< -o $@
MS_Hybrid_static: mandelbrot_set.cc Makefile
	$(MPICXX) $(COMMONFLAGS) -fopenmp \
		-DMBS_DYNAMIC=0 -DMBS_USE_OMP=1 -DMBS_USE_MPI=1 \
		$< -o $@
MS_Hybrid_dynamic: mandelbrot_set.cc Makefile
	$(MPICXX) $(COMMONFLAGS) -fopenmp \
		-DMBS_DYNAMIC=1 -DMBS_USE_OMP=1 -DMBS_USE_MPI=1 \
		$< -o $@
MS_SEQ_base: mandelbrot_set.cc Makefile
	$(CXX) $(CXXFLAGS) $(LDFLAGS) \
		-DMBS_FIXED_POINT=0 -DMBS_BLACK_HOLE=0 -DMBS_DYNAMIC=0 -DMBS_USE_OMP=0 -DMBS_USE_MPI=0 \
		$< -o $@
MS_SEQ_math: mandelbrot_set.cc Makefile
	$(CXX) $(CXXFLAGS) $(LDFLAGS) \
		-DMBS_FIXED_POINT=0 -DMBS_BLACK_HOLE=1 -DMBS_DYNAMIC=0 -DMBS_USE_OMP=0 -DMBS_USE_MPI=0 \
		$< -o $@
MS_SEQ_fixed: mandelbrot_set.cc Makefile
	$(CXX) $(CXXFLAGS) $(LDFLAGS) \
		-DMBS_FIXED_POINT=1 -DMBS_BLACK_HOLE=0  -DMBS_DYNAMIC=0 -DMBS_USE_OMP=0 -DMBS_USE_MPI=0 \
		$< -o $@

.PHONY: sources
sources: $(AUTOSOURCECODE)
MS_SEQ.cc: mandelbrot_set.cc Makefile
	unifdef $(MBSFLAGS) -DMBS_DYNAMIC=0 -DMBS_USE_OMP=0 -DMBS_USE_MPI=0 $< | clang-format > $@
MS_OpenMP_static.cc: mandelbrot_set.cc Makefile
	unifdef $(MBSFLAGS) -DMBS_DYNAMIC=0 -DMBS_USE_OMP=1 -DMBS_USE_MPI=0 $< | clang-format > $@
MS_OpenMP_dynamic.cc: mandelbrot_set.cc Makefile
	unifdef $(MBSFLAGS) -DMBS_DYNAMIC=1 -DMBS_USE_OMP=1 -DMBS_USE_MPI=0 $< | clang-format > $@
MS_MPI_static.cc: mandelbrot_set.cc Makefile
	unifdef $(MBSFLAGS) -DMBS_DYNAMIC=0 -DMBS_USE_OMP=0 -DMBS_USE_MPI=1 $< | clang-format > $@
MS_MPI_dynamic.cc: mandelbrot_set.cc Makefile
	unifdef $(MBSFLAGS) -DMBS_DYNAMIC=1 -DMBS_USE_OMP=0 -DMBS_USE_MPI=1 $< | clang-format > $@
MS_Hybrid_static.cc: mandelbrot_set.cc Makefile
	unifdef $(MBSFLAGS) -DMBS_DYNAMIC=0 -DMBS_USE_OMP=1 -DMBS_USE_MPI=1 $< | clang-format > $@
MS_Hybrid_dynamic.cc: mandelbrot_set.cc Makefile
	unifdef $(MBSFLAGS) -DMBS_DYNAMIC=1 -DMBS_USE_OMP=1 -DMBS_USE_MPI=1 $< | clang-format > $@
