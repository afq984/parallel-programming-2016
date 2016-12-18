#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct Vertex {
    std::vector<unsigned> targets;
    std::vector<unsigned> weights;
    void add_edge(unsigned dst, unsigned weight) {
        targets.push_back(dst);
        weights.push_back(weight);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm gcomm;
    assert(argc == 5);
    auto num_threads = std::stoul(argv[1]);
    auto origin = std::stoul(argv[4]) - 1;
    unsigned int num_vertices, num_edges;
    if (rank == 0) {
        std::ifstream fin(argv[2]);
        if (not fin) {
            throw std::runtime_error("bad input file");
        }
        fin >> num_vertices >> num_edges;
        std::vector<Vertex> vertices(num_vertices);
        for (unsigned int i = 0; i < num_edges; ++i) {
            unsigned u, v, w;
            fin >> u >> v >> w;
            --u;
            --v;
            vertices[u].add_edge(v, w);
            vertices[v].add_edge(u, w);
        }
        std::vector<int> nodes(num_vertices);
        std::iota(nodes.begin(), nodes.end(), 0);
        std::vector<int> degrees;
        degrees.reserve(num_vertices);
        std::vector<int> targets;
        targets.reserve(num_edges);
        std::vector<int> weights;
        weights.reserve(num_edges);
        for (const auto& vertex : vertices) {
            degrees.push_back(vertex.targets.size());
            std::copy(vertex.targets.begin(), vertex.targets.end(),
                      std::back_inserter(targets));
            std::copy(vertex.weights.begin(), vertex.weights.end(),
                      std::back_inserter(weights));
        }
        MPI_Dist_graph_create(MPI_COMM_WORLD, num_vertices, nodes.data(),
                              degrees.data(), targets.data(), weights.data(),
                              MPI_INFO_NULL, /*reorder*/ false, &gcomm);
    } else {
        MPI_Dist_graph_create(MPI_COMM_WORLD, 0, 0, 0, 0, 0, MPI_INFO_NULL,
                              /*reorder*/ false, &gcomm);
    }
    int grank;
    MPI_Comm_rank(gcomm, &grank);
    assert(rank == grank);
    int inneighbors, outneighbors, weighted;
    MPI_Dist_graph_neighbors_count(gcomm, &inneighbors, &outneighbors,
                                   &weighted);
    std::vector<int> sources(inneighbors);
    std::vector<int> sourceweights(inneighbors);
    std::vector<int> destinations(outneighbors);
    std::vector<int> destweights(outneighbors);
    MPI_Dist_graph_neighbors(gcomm, inneighbors, sources.data(),
                             sourceweights.data(), outneighbors,
                             destinations.data(), destweights.data());
    std::vector<int> neighbor_costs(inneighbors);
    int global_modified = true;
    int local_cost = grank == origin ? 0 : std::numeric_limits<int>::max();
    int local_parent = grank == origin ? origin : -1;
    while (global_modified) {
        int local_modified = false;
        MPI_Neighbor_allgather(&local_cost, 1, MPI_INT, neighbor_costs.data(),
                               1, MPI_INT, gcomm);
        for (int i = 0; i < inneighbors; ++i) {
            if (neighbor_costs[i] != std::numeric_limits<int>::max() and
                neighbor_costs[i] + sourceweights[i] < local_cost) {
                local_cost = neighbor_costs[i] + sourceweights[i];
                local_parent = sources[i];
                local_modified = true;
            }
        }
        MPI_Allreduce(&local_modified, &global_modified, 1, MPI_INT, MPI_LOR,
                      MPI_COMM_WORLD);
    }
    if (rank == 0) {
        std::vector<int> parents(num_vertices);
        MPI_Gather(&local_parent, 1, MPI_INT, parents.data(), 1, MPI_INT, 0,
                   MPI_COMM_WORLD);
        std::ofstream fout(argv[3]);
        if (not fout) {
            throw std::runtime_error("bad output file");
        }
        for (unsigned i = 0; i < num_vertices; ++i) {
            std::vector<unsigned> path;
            int cur = i;
            path.push_back(i + 1);
            do {
                cur = parents[cur];
                path.push_back(cur + 1);
            } while (cur != origin);
            std::copy(path.crbegin(), path.crend(),
                      std::ostream_iterator<unsigned>(fout, " "));
            fout << "\n";
        }
    } else {
        MPI_Gather(&local_parent, 1, MPI_INT, 0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}
