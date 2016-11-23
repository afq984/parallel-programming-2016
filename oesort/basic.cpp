#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

#include <mpi.h>

template <class T, class U>
T convert(const U& value) {
    std::stringstream ss;
    ss << value;
    T res;
    ss >> res;
    return res;
}

enum tag {
    tag_zero,
    tag_left,
    tag_right,
};

struct OE {
    const ssize_t rank;
    const ssize_t ranks;
    const ssize_t total_size;
    const char* input_file;
    const char* output_file;

    ssize_t div_;
    ssize_t mod;
    ssize_t offset;
    ssize_t size;
    ssize_t leftsize;
    ssize_t rightsize;
    int* buf0;

    OE(ssize_t rank, ssize_t ranks, ssize_t total_size, const char* input_file,
       const char* output_file)
        : rank(rank),
          ranks(ranks),
          total_size(total_size),
          input_file(input_file),
          output_file(output_file) {
        div_ = total_size / ranks;
        mod = total_size % ranks;
        offset = div_ * rank + std::min(rank, mod);
        size = div_ + (rank < mod);
        leftsize = size and rank ? (div_ + (rank - 1 < mod)) : 0;
        rightsize =
            size and (rank + 1 != ranks) ? (div_ + (rank + 1 < mod)) : 0;

        buf0 = (new int[size + 2]) + 1;
    }
    ~OE() { delete[](buf0 - 1); }

    void read_input_file() {
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY,
                      MPI_INFO_NULL, &fh);
        MPI_File_read_at_all(fh, offset * sizeof(int), buf0, size, MPI_INT,
                             MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    }

    void write_output_file() {
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, output_file,
                      MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
        MPI_File_write_at_all(fh, offset * sizeof(int), buf0, size, MPI_INT,
                              MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    }

    template <int oddeven>
    bool iteration() {
        bool converged = true;
        MPI_Request req_left, req_right;
        const bool do_left = (offset + oddeven) % 2 and leftsize;
        const bool do_right = (offset + size + oddeven) % 2 and rightsize;
        if (do_left) {
            MPI_Irecv(buf0 - 1, 1, MPI_INT, rank - 1, tag_right, MPI_COMM_WORLD,
                      &req_left);
        }
        if (do_right) {
            MPI_Irecv(buf0 + size, 1, MPI_INT, rank + 1, tag_left,
                      MPI_COMM_WORLD, &req_right);
        }
        if (do_left) {
            MPI_Send(buf0, 1, MPI_INT, rank - 1, tag_left, MPI_COMM_WORLD);
        }
        if (do_right) {
            MPI_Send(buf0 + size - 1, 1, MPI_INT, rank + 1, tag_right,
                     MPI_COMM_WORLD);
        }
        if (do_left) {
            MPI_Wait(&req_left, MPI_STATUS_IGNORE);
        }
        if (do_right) {
            MPI_Wait(&req_right, MPI_STATUS_IGNORE);
        }
        register ssize_t i;
        ssize_t term;
        if ((oddeven + offset) % 2) {
            if (do_left) {
                i = -1;
            } else {
                i = 1;
            }
        } else {
            i = 0;
        }
        if (do_right) {
            term = size;
        } else {
            term = size - 1;
        }
        for (; i < term; i += 2) {
            if (buf0[i] > buf0[i + 1]) {
                converged = false;
                std::swap(buf0[i], buf0[i + 1]);
            }
        }
        return converged;
    }

    void sort() {
        read_input_file();
        bool global_converged = false;
        while (not global_converged) {
            bool converged = true;
            converged &= iteration<0>();
            converged &= iteration<1>();
            MPI_Allreduce(&converged, &global_converged, 1, MPI::BOOL, MPI_LAND,
                          MPI_COMM_WORLD);
        }
        write_output_file();
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " N input output\n";
        return -1;
    }

    OE oe(rank, ranks, convert<ssize_t>(argv[1]), argv[2], argv[3]);
    oe.sort();

    MPI_Finalize();
}
