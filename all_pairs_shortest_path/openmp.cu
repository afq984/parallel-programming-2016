#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <fstream>

#define XB 24
const int INF = 10000000;

int n, m;  // Number of vertices, edges
size_t devPitch[2], hostPitch;
unsigned* hostPtr;

#define check(err) __check(err, __LINE__, #err)
template <typename T>
void __check(cudaError err, int line, T source) {
    if (err) {
        fprintf(stderr, "%d:%s\n%s\n", line, cudaGetErrorString(err), source);
        abort();
    }
}

#define dist(i, j) (hostPtr[(i)*n + (j)])

void sincelast(const char* message = 0) {
    static auto last = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%16s  %lf\n", message,
            ((std::chrono::duration<double>)(now - last)).count());
    last = now;
}

void input(char* inFileName) {
    FILE* infile = fopen(inFileName, "r");
    fscanf(infile, "%d %d", &n, &m);

    hostPitch = sizeof(int) * n;
    check(cudaMallocHost(&hostPtr, hostPitch * n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                dist(i, j) = 0;
            else
                dist(i, j) = INF;
        }
    }

    while (--m >= 0) {
        int a, b, v;
        fscanf(infile, "%u%u%u", &a, &b, &v);
        dist(a - 1, b - 1) = v;
    }
}

void output(char* outFileName) {
    std::ofstream fout(outFileName);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (dist(i, j) >= INF)
                fout << "INF ";
            else
                fout << dist(i, j) << " ";
        }
        fout << "\n";
    }
}

#define safeat(i, j) ((i) < n and (j) < n ? at(i, j) : INF)
#define at(i, j) (((int*)((char*)devPtr + (i)*devPitch))[j])
#define mifn(p, q)                \
    {                             \
        if ((p) > (q)) (p) = (q); \
    }

__global__ void phase1(unsigned* devPtr, int r, int n, int kmin, int devPitch) {
    int i = kmin + threadIdx.y;
    int j = kmin + threadIdx.x;
    if (i >= n or j >= n) {
        return;
    }
    int kmax = kmin + blockDim.x < n ? blockDim.x : n - kmin;
    __shared__ unsigned pivot[XB][XB];
    pivot[threadIdx.y][threadIdx.x] = at(i, j);
    for (int k = 0; k < kmax; ++k) {
        __syncthreads();
        mifn(pivot[threadIdx.y][threadIdx.x],
             pivot[threadIdx.y][k] + pivot[k][threadIdx.x]);
    }
    at(i, j) = pivot[threadIdx.y][threadIdx.x];
}

__global__ void phase2a(unsigned* devPtr, int r, int n, int kmin,
                        int devPitch) {
    if (blockIdx.y == r) {
        return;
    }
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = r * blockDim.x + threadIdx.x;
    int kmax = kmin + blockDim.x < n ? blockDim.x : n - kmin;
    __shared__ unsigned pivot[XB][XB], block[XB][XB];
    pivot[threadIdx.y][threadIdx.x] =
        safeat(kmin + threadIdx.y, kmin + threadIdx.x);
    block[threadIdx.y][threadIdx.x] = safeat(i, j);
    if (i >= n or j >= n) {
        return;
    }
    for (int k = 0; k < kmax; ++k) {
        __syncthreads();
        mifn(block[threadIdx.y][threadIdx.x],
             block[threadIdx.y][k] + pivot[k][threadIdx.x]);
    }
    at(i, j) = block[threadIdx.y][threadIdx.x];
}

__global__ void phase2b(unsigned* devPtr, int r, int n, int kmin,
                        int devPitch) {
    if (blockIdx.x == r) {
        return;
    }
    int i = r * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int kmax = kmin + blockDim.x < n ? blockDim.x : n - kmin;
    __shared__ unsigned pivot[XB][XB], block[XB][XB];
    pivot[threadIdx.y][threadIdx.x] =
        safeat(kmin + threadIdx.y, kmin + threadIdx.x);
    block[threadIdx.y][threadIdx.x] = safeat(i, j);
    if (i >= n or j >= n) {
        return;
    }
    for (int k = 0; k < kmax; ++k) {
        __syncthreads();
        mifn(block[threadIdx.y][threadIdx.x],
             block[k][threadIdx.x] + pivot[threadIdx.y][k]);
    }
    at(i, j) = block[threadIdx.y][threadIdx.x];
}

__global__ void phase3(unsigned* devPtr, int r, int n, int kmin, int devPitch,
                       int ioff) {
    if (blockIdx.x == r or blockIdx.y + ioff == r) {
        return;
    }
    int i = (blockIdx.y + ioff) * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int kmax = kmin + blockDim.x < n ? blockDim.x : n - kmin;
    __shared__ unsigned iref[XB][XB], jref[XB][XB];
    iref[threadIdx.y][threadIdx.x] = safeat(i, kmin + threadIdx.x);
    jref[threadIdx.y][threadIdx.x] = safeat(kmin + threadIdx.y, j);
    if (i >= n or j >= n) {
        return;
    }
    int local = at(i, j);
    __syncthreads();
    for (int k = 0; k < kmax; ++k) {
        mifn(local, iref[threadIdx.y][k] + jref[k][threadIdx.x]);
    }
    at(i, j) = local;
}

#undef at

int ceil(int a, int b) { return (a + b - 1) / b; }

void blockfw(int B) {
    unsigned* devPtr[2];
    cudaStream_t streams[2];
    for (int i = 0; i < 2; ++i) {
        check(cudaStreamCreate(&streams[i]));
        check(cudaSetDevice(i));
        check(cudaMallocPitch(&devPtr[i], &devPitch[i], hostPitch, n));
        check(cudaMemcpy2D(devPtr[i], devPitch[i], hostPtr, hostPitch,
                           hostPitch, n, cudaMemcpyHostToDevice));
    }
    sincelast("hostToDevice");
    int rounds = ceil(n, B);
    int rdown = rounds / 2;
    int rup = (rounds + 1) / 2;
#pragma omp parallel num_threads(2)
    {
        const int i = omp_get_thread_num();
        check(cudaSetDevice(i));
        check(cudaMallocPitch(&devPtr[i], &devPitch[i], hostPitch, n));
        check(cudaMemcpy2D(devPtr[i], devPitch[i], hostPtr, hostPitch,
                           hostPitch, n, cudaMemcpyHostToDevice));
        for (int r = 0; r < rounds; ++r) {
            int kmin = r * B;
            int bnext = kmin + B + B < n ? B : n - B - kmin;
            if (i == 0) {
                phase1<<<1, dim3(B, B)>>>(devPtr[0], r, n, kmin, devPitch[0]);
                phase2a<<<dim3(1, rounds), dim3(B, B)>>>(devPtr[0], r, n, kmin,
                                                         devPitch[0]);
                phase2b<<<dim3(rounds, 1), dim3(B, B)>>>(devPtr[0], r, n, kmin,
                                                         devPitch[0]);
                phase3<<<dim3(rounds, rup), dim3(B, B)>>>(devPtr[0], r, n, kmin,
                                                          devPitch[0], rdown);
                if (r + 1 < rounds) {
                    check(cudaMemcpy2D(
                        (char*)devPtr[1] + devPitch[1] * rdown * B +
                            sizeof(unsigned) * (kmin + B),
                        devPitch[1],
                        (char*)devPtr[0] + devPitch[0] * rdown * B +
                            sizeof(unsigned) * (kmin + B),
                        devPitch[0], bnext, n - rdown * B,
                        cudaMemcpyDeviceToDevice));
                    if (r + 1 >= rdown) {
                        check(cudaMemcpy2D(
                            (char*)devPtr[1] + devPitch[1] * (kmin + B),
                            devPitch[1],
                            (char*)devPtr[0] + devPitch[0] * (kmin + B),
                            devPitch[0], hostPitch, bnext,
                            cudaMemcpyDeviceToDevice));
                    }
                }
            } else if (rdown) {
                check(cudaSetDevice(1));
                phase1<<<1, dim3(B, B)>>>(devPtr[1], r, n, kmin, devPitch[1]);
                phase2a<<<dim3(1, rounds), dim3(B, B)>>>(devPtr[1], r, n, kmin,
                                                         devPitch[1]);
                phase2b<<<dim3(rounds, 1), dim3(B, B)>>>(devPtr[1], r, n, kmin,
                                                         devPitch[1]);
                phase3<<<dim3(rounds, rdown), dim3(B, B)>>>(
                    devPtr[1], r, n, kmin, devPitch[1], 0);
                if (r + 1 < rounds) {
                    check(cudaMemcpy2D(devPtr[0] + kmin + B, devPitch[0],
                                       devPtr[1] + kmin + B, devPitch[1], bnext,
                                       rdown * B, cudaMemcpyDeviceToDevice));
                    if (r + 1 < rdown) {
                        check(cudaMemcpy2D(
                            (char*)devPtr[0] + devPitch[0] * (kmin + B),
                            devPitch[0],
                            (char*)devPtr[1] + devPitch[1] * (kmin + B),
                            devPitch[1], hostPitch, bnext,
                            cudaMemcpyDeviceToDevice));
                    }
                }
            }
#pragma omp barrier
        }
        // check(cudaMemcpy2D(hostPtr, hostPitch, devPtr[1],
        // devPitch[1],
        //                    hostPitch, rdown * B,
        //                    cudaMemcpyDeviceToHost));
        // check(cudaSetDevice(0));
        // check(cudaMemcpy2D(
        //     (char*)hostPtr + hostPitch * rdown * B, hostPitch,
        //     (char*)devPtr[0] + devPitch[0] * rdown * B, devPitch[1],
        //     hostPitch, n - rdown * B, cudaMemcpyDeviceToHost));
        // check(cudaMemcpy2D(devPtr[0], devPitch[0], hostPtr,
        // hostPitch,
        //                    hostPitch, rdown * B,
        //                    cudaMemcpyHostToDevice));
        // check(cudaSetDevice(1));
        // check(cudaMemcpy2D(
        //     (char*)devPtr[1] + devPitch[1] * rdown * B, devPitch[0],
        //     (char*)hostPtr + hostPitch * rdown * B, hostPitch,
        //     hostPitch,
        //     n - rdown * B, cudaMemcpyHostToDevice));
    }

    check(cudaDeviceSynchronize());
    sincelast("compute");
    check(cudaGetLastError());

    check(cudaMemcpy2D(hostPtr, hostPitch, devPtr[1], devPitch[1], hostPitch,
                       rdown * B, cudaMemcpyDeviceToHost));
    check(cudaMemcpy2D((char*)hostPtr + hostPitch * rdown * B, hostPitch,
                       (char*)devPtr[0] + devPitch[0] * rdown * B, devPitch[1],
                       hostPitch, n - rdown * B, cudaMemcpyDeviceToHost));
    sincelast("deviceToHost");
}

int main(int argc, char* argv[]) {
    sincelast("placeholder");
    int B = atoi(argv[3]);
    input(argv[1]);
    mifn(B, XB);
    sincelast("input");
    blockfw(B);
    output(argv[2]);
    sincelast("output");
}
