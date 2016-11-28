#if MBS_USE_MPI
#include <mpi.h>
#endif
#if MBS_USE_OMP
#include <omp.h>
#endif
#include <X11/Xlib.h>
#include <stdint.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if MBS_FIXED_POINT
#define HALF 26
#define FULL 52
#endif

#if MBS_USE_OMP && MBS_DYNAMIC
#ifndef MBS_OMP_CHUNK_SIZE
#define MBS_OMP_CHUNK_SIZE 32
#endif
#else
#undef MBS_OMP_CHUNK_SIZE
#define MBS_OMP_CHUNK_SIZE 0
#endif
#if MBS_USE_MPI && MBS_DYNAMIC
#ifndef MBS_MPI_MASTER_CHUNK_SIZE
#define MBS_MPI_MASTER_CHUNK_SIZE 64
#endif
#ifndef MBS_MPI_SLAVE_CHUNK_SIZE
#define MBS_MPI_SLAVE_CHUNK_SIZE pixels / ranks / 32
#endif
#else
#undef MBS_MPI_MASTER_CHUNK_SIZE
#undef MBS_MPI_SLAVE_CHUNK_SIZE
#define MBS_MPI_MASTER_CHUNK_SIZE 0
#define MBS_MPI_SLAVE_CHUNK_SIZE 0
#endif
#ifndef MBS_STATIC_INTERLEAVING
#define MBS_STATIC_INTERLEAVING 1
#endif

#define SQ(x) ((x) * (x))

template <class T>
T convert(const char* str) {
    std::istringstream iss(str);
    T to;
    if (iss >> to) {
        return to;
    } else {
        throw std::runtime_error(std::string("invalid parameter ") + str);
    }
}

enum tag {
    t_unused,
    t_request,
    t_job,
    t_result,
};

int main(int argc, char** argv) {
    // pre-initialize
    auto start = std::chrono::high_resolution_clock::now();
#if MBS_USE_MPI
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
#define ranks 1
#endif

// parameter parsing
#if MBS_USE_OMP
    const auto num_threads = convert<int>(argv[1]);
    omp_set_num_threads(num_threads);
#else
#define num_threads 1
#endif
    const auto left = convert<double>(argv[2]);
    const auto right = convert<double>(argv[3]);
    const auto lower = convert<double>(argv[4]);
    const auto upper = convert<double>(argv[5]);
    const auto width = convert<int>(argv[6]);
    const auto height = convert<int>(argv[7]);
    const auto enable_display = argv[8] == std::string("enable");
    const auto pixels = width * height;

// MPI preparation
#if !(MBS_USE_MPI)  // no MPI
#define local_offset 0
#define local_pixels pixels
#define master_arr arr
    auto arr = new unsigned char[pixels];
#else
    unsigned char* arr = 0;
    unsigned char* master_arr = 0;
#if !(MBS_DYNAMIC)  // MPI static
    master_arr = new unsigned char[rank == 0 ? pixels : 0];

    // set workload size
    auto recvcounts = new int[ranks];
    auto displs = new int[ranks];
    const int _div = pixels / ranks;
    const int _mod = pixels % ranks;
    recvcounts[0] = (_div + (_mod != 0));
    displs[0] = 0;
    for (int r = 0; r < ranks; ++r) {
        recvcounts[r] = (_div + (r < _mod));
        displs[r] = displs[r - 1] + recvcounts[r - 1];
    }
#if !MBS_STATIC_INTERLEAVING
    const int local_offset = displs[rank];
#endif
    const int local_pixels = recvcounts[rank];

    arr = new unsigned char[local_pixels];
#else               // MBS_DYNAMIC
    MPI_Request* req_requests = 0;
    int ava_count;
    int* available = 0;
    MPI_Request work_request;
    std::vector<MPI_Request> result_requests;
    const int _c_MBS_MPI_SLAVE_CHUNK_SIZE =
        std::max(1024, MBS_MPI_SLAVE_CHUNK_SIZE);
    int local_offset;
    const int local_pixels =
        rank == 0 ? MBS_MPI_MASTER_CHUNK_SIZE : _c_MBS_MPI_SLAVE_CHUNK_SIZE;
    int master_offset = 0;

    if (rank == 0) {
        master_arr = new unsigned char[pixels + _c_MBS_MPI_SLAVE_CHUNK_SIZE];
        req_requests = new MPI_Request[ranks];
        available = new int[ranks];
        for (int r = 0; r < ranks; ++r) {
            MPI_Irecv(0, 0, MPI_INT, r, t_request, MPI_COMM_WORLD,
                      req_requests + r);
        }
    } else {
        arr = new unsigned char[_c_MBS_MPI_SLAVE_CHUNK_SIZE];
    }
#endif
#endif

#if MBS_USE_OMP
#pragma omp parallel
    {
#endif

#if MBS_USE_MPI && MBS_DYNAMIC
        while (true) {
#if MBS_USE_OMP
#pragma omp single
            {
#endif
                // determine rank workload
                if (rank == 0) {
                    arr = master_arr + master_offset;
                    local_offset = master_offset;
                    master_offset += MBS_MPI_MASTER_CHUNK_SIZE;
                    MPI_Testsome(ranks, req_requests, &ava_count, available,
                                 MPI_STATUSES_IGNORE);
                    for (int a = 0; a < ava_count and master_offset < pixels;
                         ++a) {
                        auto r = available[a];
                        MPI_Send(&master_offset, 1, MPI_INT, r, t_job,
                                 MPI_COMM_WORLD);
                        result_requests.emplace_back();
                        MPI_Irecv(master_arr + master_offset,
                                  _c_MBS_MPI_SLAVE_CHUNK_SIZE, MPI_CHAR, r,
                                  t_result, MPI_COMM_WORLD,
                                  &result_requests.back());
                        MPI_Irecv(0, 0, MPI_INT, r, t_request, MPI_COMM_WORLD,
                                  req_requests + r);
                        master_offset += _c_MBS_MPI_SLAVE_CHUNK_SIZE;
                    }
                } else {
                    MPI_Send(0, 0, MPI_INT, 0, t_request, MPI_COMM_WORLD);
                    MPI_Recv(&local_offset, 1, MPI_INT, 0, t_job,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
#if MBS_USE_OMP
            }  // omp single
#pragma omp barrier
#endif
            if (local_offset >= pixels) {
                break;
            }

#endif

#if MBS_USE_OMP
#if !(MBS_DYNAMIC)
// static OpenMP
#if MBS_STATIC_INTERLEAVING
#pragma omp for schedule(static, 1) nowait
#else
#pragma omp for schedule(static) nowait
#endif
#else
// dynamic OpenMP
#pragma omp for schedule(dynamic, MBS_OMP_CHUNK_SIZE) nowait
#endif
#endif

            for (int p = 0; p < local_pixels; ++p) {
#if !MBS_USE_MPI || MBS_DYNAMIC || !MBS_STATIC_INTERLEAVING
                const int i = (p + local_offset) / height;
                const int j = (p + local_offset) % height;
#else
        const int i = ((p * ranks) + rank) / height;
        const int j = ((p * ranks) + rank) % height;
#endif

                double xd = i * ((right - left) / width) + left;
                double yd = j * ((upper - lower) / height) + lower;

#if MBS_FIXED_POINT
                if (xd < -2 or xd > 2 or yd < -2 or yd > 2) {
                    arr[p] = 1;
                    continue;
                }
#endif
#if MBS_BLACK_HOLE
                if ((xd + 1) * (xd + 1) + yd * yd < 0.0625) {
                    arr[p] = 0;
                    continue;
                }
                if (SQ(SQ(xd - .25) + SQ(yd) + 0.5 * (xd - .25)) <
                    0.25 * (SQ(xd - .25) + SQ(yd))) {
                    arr[p] = 0;
                    continue;
                }
#endif

                // actual calculation
                int repeats = 0;
#if MBS_FIXED_POINT
                int64_t x0 = xd * (1l << FULL);
                int64_t y0 = yd * (1l << FULL);
                int64_t xt = 0;
                int64_t yt = 0;
                int64_t lengthsq = 0;

                while (repeats < 100000 && (lengthsq >> FULL) < 4) {
                    int64_t temp = SQ(xt >> HALF) - SQ(yt >> HALF) + x0;
                    yt = 2 * ((xt >> HALF) * (yt >> HALF)) + y0;
                    xt = temp;
                    lengthsq = SQ(xt >> HALF) + SQ(yt >> HALF);
                    ++repeats;
                }
#else
        double xt = 0;
        double yt = 0;
        double lengthsq = 0;
        while (repeats < 100000 && lengthsq < 4) {
            double temp = SQ(xt) - SQ(yt) + xd;
            yt = 2 * (xt * yt) + yd;
            xt = temp;
            lengthsq = SQ(xt) + SQ(yt);
            ++repeats;
        }
#endif
                arr[p] = repeats % 256;
            }  // for

#if MBS_USE_MPI && MBS_DYNAMIC  // MPI requires sort of cleanup after loop

#if MBS_USE_OMP
#pragma omp barrier
#pragma omp single
#endif
            if (rank != 0) {
                MPI_Send(arr, _c_MBS_MPI_SLAVE_CHUNK_SIZE, MPI_CHAR, 0,
                         t_result, MPI_COMM_WORLD);
            }
        }  // MPI DYNAMIC while loop
#endif

#if MBS_USE_OMP
    }  // omp parallel
#endif

#if MBS_USE_MPI
#if !(MBS_DYNAMIC)
    MPI_Gatherv(arr, local_pixels, MPI_CHAR, master_arr, recvcounts, displs,
                MPI_CHAR, 0, MPI_COMM_WORLD);
#else
    if (rank == 0) {
        for (int r = 1; r < ranks; ++r) {
            MPI_Isend(&master_offset, 1, MPI_INT, r, t_job, MPI_COMM_WORLD,
                      &work_request);
        }
        MPI_Waitall(result_requests.size(), result_requests.data(),
                    MPI_STATUSES_IGNORE);
    }
#endif
    // timing & drawing
    if (rank == 0) {
#endif

        auto stop = std::chrono::high_resolution_clock::now();
        // FixedPoint | BlackHole | OMP | Threads | OMP Chunk | MPI | Processes
        // | Master Chunk | Slave Chunk | Width | Height | Time
        std::cout << MBS_FIXED_POINT << '\t' << MBS_BLACK_HOLE << '\t'
                  << MBS_USE_OMP << '\t' << num_threads << '\t'
                  << MBS_OMP_CHUNK_SIZE << '\t' << MBS_USE_MPI << '\t' << ranks
                  << '\t' << MBS_MPI_MASTER_CHUNK_SIZE << '\t'
                  << MBS_MPI_SLAVE_CHUNK_SIZE << '\t' << width << '\t' << height
                  << '\t'
                  << ((std::chrono::duration<double>)(stop - start)).count()
                  << std::endl;
        if (enable_display) {
            Display* display;
            Window window;  // initialization for a window
            int screen;     // which screen

            GC gc;
            XGCValues values;
            display = XOpenDisplay(NULL);
            if (display == NULL) {
                std::cerr << "cannot open display\n";
                return 1;
            }

            screen = DefaultScreen(display);

            /* create window */
            window = XCreateSimpleWindow(
                display, RootWindow(display, screen), /*x*/ 0,
                /*y*/ 0, width, height, /*border_width*/ 0,
                BlackPixel(display, screen), WhitePixel(display, screen));

            gc = XCreateGC(display, window, /*valuemask*/ 0, &values);

            XMapWindow(display, window);
            XSync(display, 0);
#if !MBS_USE_MPI || MBS_DYNAMIC || !MBS_STATIC_INTERLEAVING
            for (int i = 0; i < width; ++i) {
                for (int j = 0; j < height; ++j) {
                    XSetForeground(display, gc, master_arr[i * height + j]
                                                    << 20);
                    XDrawPoint(display, window, gc, i, j);
                }
            }
#else
        for (int r = 0; r < ranks; ++r) {
            for (int p = 0; p < recvcounts[r]; ++p) {
                XSetForeground(display, gc, master_arr[displs[r] + p] << 20);
                XDrawPoint(display, window, gc, (p * ranks + r) / height,
                           (p * ranks + r) % height);
            }
        }
#endif
            XFlush(display);
            sleep(5);
        }

#if MBS_USE_MPI
    }  // if rank == 0
    MPI_Finalize();
#endif
}
