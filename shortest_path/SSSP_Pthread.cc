#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <boost/heap/fibonacci_heap.hpp>

struct Edge {
    const int target;
    const int cost;
    Edge(int target, int cost) : target(target), cost(cost) {}
};

struct Vertex {
    std::mutex mutex;
    std::vector<Edge> edges;
    void add_edge(int target, int cost) {
        std::lock_guard<std::mutex> lock(mutex);
        edges.emplace_back(target, cost);
    }
};

struct DistInfo {
    int v;
    int cost = std::numeric_limits<int>::max();
    int parent = -1;
    bool operator<(const DistInfo& that) const {
        return this->cost > that.cost;
    }
};

struct FilePartition {
    const int threads;
    const ssize_t start, stop, diff, div, mod;
    FilePartition(int threads, ssize_t start, ssize_t stop)
        : threads(threads),
          start(start),
          stop(stop),
          diff(stop - start),
          div(diff / threads),
          mod(diff % threads) {}
    ssize_t operator[](ssize_t tid) const {
        return start + div * tid + std::min(mod, tid);
    }
};

struct IntFile {
    std::ifstream ifs;
    int c = ' ';
    ssize_t g = 0;
    IntFile(const char* filename) : ifs(filename) {}
    operator bool() { return (bool)ifs; }
    IntFile& operator>>(int& other) {
        other = 0;
        while (c == ' ' or c == '\n' or c == '\t') {
            c = ifs.get();
            ++g;
        }
        while ('0' <= c and c <= '9') {
            other *= 10;
            other += c - '0';
            c = ifs.get();
            ++g;
        }
        return *this;
    }
    void seekg(ssize_t pos) {
        ifs.seekg(pos);
        g = pos;
    }
    ssize_t tellg() { return g; }
    void ignore(ssize_t, int chr) {
        while (c != '\n' and c != EOF) {
            ++g;
            c = ifs.get();
        }
    }
};

char* infile;
std::vector<Vertex> vertices;

void read_file(ssize_t from, ssize_t to) {
    // printf("%ld %ld %ld %s\n", from, to, vertices.size(), infile);
    IntFile fin(infile);
    assert(fin);
    fin.seekg(from);
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    while (fin.tellg() < to) {
        int u, v, cost;
        fin >> u >> v >> cost;
        fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        --u;
        --v;
        // printf("%d %d %d\n", u, v, cost);
        vertices.at(u).add_edge(v, cost);
        vertices.at(v).add_edge(u, cost);
    }
}

void sincelast(const char* message = 0) {
    static auto last = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    std::cout << message
              << ((std::chrono::duration<double>)(now - last)).count()
              << std::endl;
    last = now;
}

int main(int argc, char** argv) {
    std::ios_base::sync_with_stdio(false);
    assert(argc == 5);
    auto num_threads = std::stoi(argv[1]);
    assert(num_threads > 0);
    infile = argv[2];
    std::ifstream fin(argv[2]);
    std::ofstream fout(argv[3]);
    auto origin = std::stoi(argv[4]) - 1;
    if (not fin) {
        throw std::runtime_error("bad input file");
    }
    if (not fout) {
        throw std::runtime_error("bad output file");
    }
    int num_vertices, num_edges;
    fin >> num_vertices >> num_edges;
    auto file_start = fin.tellg();
    fin.seekg(0, std::ios_base::end);
    auto file_end = fin.tellg();
    FilePartition fp(num_threads, file_start, file_end);
    vertices = std::vector<Vertex>(num_vertices);
    std::vector<std::thread> rfthreads;
    sincelast("init ");
    for (int i = 0; i < num_threads; ++i) {
        rfthreads.emplace_back(read_file, fp[i] - 1, fp[i + 1]);
    }
    for (auto& thread : rfthreads) {
        thread.join();
    }
    sincelast("input ");
    std::vector<DistInfo> distinfo(num_vertices);
    distinfo[origin].cost = 0;
    distinfo[origin].parent = origin;
    using heap_t = boost::heap::fibonacci_heap<DistInfo>;
    heap_t heap;
    std::vector<heap_t::handle_type> handles;
    handles.reserve(num_vertices);
    for (int v = 0; v < num_vertices; ++v) {
        distinfo[v].v = v;
        handles.emplace_back(heap.push(distinfo[v]));
    }
    while (not heap.empty()) {
        int v = heap.top().v;
        heap.pop();
        const auto& vertex = vertices[v];
        const auto& di = distinfo[v];
        for (const auto& edge : vertex.edges) {
            auto& dst = distinfo[edge.target];
            auto off = di.cost + edge.cost;
            if (dst.cost > off) {
                dst.cost = off;
                dst.parent = v;
                heap.increase(handles[edge.target], dst);
            }
        }
    }
    sincelast("compute ");
    for (int i = 0; i < num_vertices; ++i) {
        std::vector<int> path;
        int cur = i;
        path.push_back(cur + 1);
        do {
            cur = distinfo[cur].parent;
            path.push_back(cur + 1);
        } while (cur != origin);
        std::copy(path.crbegin(), path.crend(),
                  std::ostream_iterator<int>(fout, " "));
        fout << '\n';
    }
    sincelast("output ");
}
