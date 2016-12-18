ALL = SSSP_Pthread SSSP_MPI_sync SSSP_MPI_async
.PHONY: all
all: $(ALL)
.PHONY: clean
clean:
	rm -f $(ALL)
.PHONY: format
format: $(addsuffix .cc,$(ALL))
	clang-format -i $^
SSSP_Pthread: SSSP_Pthread.cc
	g++ -std=c++11 -O3 -pthread -o $@ $^
SSSP_MPI_%: SSSP_MPI_%.cc
	mpicxx -std=c++11 -O3 -o $@ $^
