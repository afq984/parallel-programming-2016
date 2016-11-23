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

void _merge_large(int* dstl, int* dstr, int* al, int* ar, int* bl, int* br) {
    --dstr;
    --ar;
    --br;
    while (dstl <= dstr and al <= ar and bl <= br) {
        if (*br < *ar) {
            *dstr-- = *ar--;
        } else {
            *dstr-- = *br--;
        }
    }
    if (dstl <= dstr) {
        if (al <= ar) {
            int remaining = std::min(dstr - dstl, ar - al) + 1;
            memcpy(dstl, ar - remaining + 1, remaining * sizeof(int));
        } else {
            int remaining = std::min(dstr - dstl, br - bl) + 1;
            memcpy(dstl, br - remaining + 1, remaining * sizeof(int));
        }
    }
}

void merge_large(int* dstl, ssize_t dstn, int* al, ssize_t an, int* bl,
                 ssize_t bn) {
    _merge_large(dstl, dstl + dstn, al, al + an, bl, bl + bn);
}

void _merge_small(int* dstl, int* dstr, int* al, int* ar, int* bl, int* br) {
    while (dstl < dstr and al < ar and bl < br) {
        if (*al < *bl) {
            *dstl++ = *al++;
        } else {
            *dstl++ = *bl++;
        }
    }
    if (dstl < dstr) {
        if (al < ar) {
            memcpy(dstl, al, sizeof(int) * std::min(dstr - dstl, ar - al));
        } else {
            memcpy(dstl, bl, sizeof(int) * std::min(dstr - dstl, br - bl));
        }
    }
}

void merge_small(int* dstl, ssize_t dstn, int* al, ssize_t an, int* bl,
                 ssize_t bn) {
    _merge_small(dstl, dstl + dstn, al, al + an, bl, bl + bn);
}

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
    int* buf1;
    int*& currbuf;
    int* foreignbuf;

    OE(ssize_t rank, ssize_t ranks, ssize_t total_size, const char* input_file,
       const char* output_file)
        : rank(rank),
          ranks(ranks),
          total_size(total_size),
          input_file(input_file),
          output_file(output_file),
          currbuf(rank % 2 ? buf1 : buf0) {
        div_ = total_size / ranks;
        mod = total_size % ranks;
        offset = div_ * rank + std::min(rank, mod);
        size = div_ + (rank < mod);
        leftsize = div_ + (rank - 1 < mod);
        rightsize = div_ + (rank + 1 < mod);

        buf0 = new int[size];
        buf1 = new int[size];
        foreignbuf = new int[leftsize];
    }

    void read_input_file() {
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY,
                      MPI_INFO_NULL, &fh);
        MPI_File_read_at_all(fh, offset * sizeof(int), currbuf, size, MPI_INT,
                             MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    }

    void write_output_file() {
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, output_file,
                      MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
        MPI_File_write_at_all(fh, offset * sizeof(int), currbuf, size, MPI_INT,
                              MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    }

    bool do_left() {
        // do one iteration with the previous rank, return true if local data
        // changed
        if (rank == 0 or size == 0) {
            std::swap(buf0, buf1);
            return false;
        }
        MPI_Request request;
        MPI_Irecv(foreignbuf, leftsize, MPI_INT, rank - 1, tag_right,
                  MPI_COMM_WORLD, &request);
        MPI_Send(buf1, size, MPI_INT, rank - 1, tag_left, MPI_COMM_WORLD);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        if (foreignbuf[leftsize - 1] <= buf1[0]) {
            std::swap(buf0, buf1);
            return false;
        }
        merge_large(buf0, size, buf1, size, foreignbuf, leftsize);
        return true;
    }

    bool do_right() {
        // do one iteration with the next rank, return true if local data
        // changed
        if (rank + 1 == ranks or size == 0 or rightsize == 0) {
            std::swap(buf0, buf1);
            return false;
        }
        MPI_Request request;
        MPI_Irecv(foreignbuf, rightsize, MPI_INT, rank + 1, tag_left,
                  MPI_COMM_WORLD, &request);
        MPI_Send(buf0, size, MPI_INT, rank + 1, tag_right, MPI_COMM_WORLD);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        if (buf0[size - 1] <= foreignbuf[0]) {
            std::swap(buf0, buf1);
            return false;
        }
        merge_small(buf1, size, buf0, size, foreignbuf, rightsize);
        return true;
    }

    void sort() {
        read_input_file();
        std::sort(currbuf, currbuf + size);
        bool global_converged = false;
        if (rank % 2) {
            while (not global_converged) {
                bool converged = true;
                converged &= not do_left();
                converged &= not do_right();
                MPI_Allreduce(&converged, &global_converged, 1, MPI::BOOL,
                              MPI_LAND, MPI_COMM_WORLD);
            }
        } else {
            while (not global_converged) {
                bool converged = true;
                converged &= not do_right();
                converged &= not do_left();
                MPI_Allreduce(&converged, &global_converged, 1, MPI::BOOL,
                              MPI_LAND, MPI_COMM_WORLD);
            }
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
