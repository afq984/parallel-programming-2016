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

enum tag { T_UNUSED, Cost, Token, Termination };

enum token { White = 0, Black = 1 };

const int zero = 0;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
    int local_cost = grank == origin ? 0 : std::numeric_limits<int>::max();
    int local_parent = grank == origin ? origin : -1;
    int ring_next = (grank + 1) % ranks;
    int ring_prev = grank == 0 ? ranks - 1 : grank - 1;
    int color = White;
    std::vector<int> distrecv(inneighbors);
    int prevcolor;
    std::vector<MPI_Request> recvreqs(inneighbors + 2);
    auto recvindex = [&](int index) {
        MPI_Irecv(distrecv.data() + index, 1, MPI_INT, sources[index], Cost,
                  gcomm, recvreqs.data() + index);
    };
    auto recvcolor = [&]() {
        MPI_Irecv(&prevcolor, 1, MPI_INT, ring_prev, Token, gcomm,
                  recvreqs.data() + inneighbors);
    };
    MPI_Irecv(0, 0, MPI_INT, MPI_ANY_SOURCE, Termination, gcomm,
              recvreqs.data() + inneighbors + 1);
    for (int i = 0; i < inneighbors; ++i) {
        recvindex(i);
    }
    recvcolor();
    auto broadcast = [&]() {
        for (int i = 0; i < outneighbors; ++i) {
            if (destinations[i] == local_parent) {
                continue;
            }
            if (i < grank) {
                color = Black;
            }
            int tosend = local_cost + destweights[i];
            MPI_Send(&tosend, 1, MPI_INT, destinations[i], Cost, gcomm);
        }
    };
    if (grank == origin) {
        broadcast();
        MPI_Send(&color, 1, MPI_INT, ring_next, Token, gcomm);
    }
    std::vector<int> some(inneighbors + 2);
    int howmany;
    bool terminated = false;
    while (not terminated) {
        MPI_Waitsome(inneighbors + 2, recvreqs.data(), &howmany, some.data(),
                     MPI_STATUSES_IGNORE);
        bool token_arrived = false;
        // fprintf(stderr, "%d %d\n", grank, howmany);
        for (int i = 0; i < howmany; ++i) {
            auto& index = some[i];
            if (index < inneighbors) {
                int distance = distrecv.at(index);
                recvindex(index);
                if (distance < local_cost) {
                    // fprintf(stderr, "%d updated by %d, %d\n", grank,
                    // sources.at(index), distance);
                    local_cost = distance;
                    local_parent = sources.at(index);
                    broadcast();
                }
            } else if (index == inneighbors) {
                token_arrived = true;
            } else {
                assert(index == inneighbors + 1);
                // printf("%d terminated\n", grank);
                terminated = true;
                for (int i = grank * 2; i < grank * 2 + 2 and i < ranks; ++i) {
                    MPI_Send(0, 0, MPI_INT, i, Termination, gcomm);
                }
                break;
            }
        }
        if (token_arrived) {
            int val = prevcolor;
            recvcolor();
            assert(val == Black or val == White);
            if (grank == 0 and val == White) {
                MPI_Send(0, 0, MPI_INT, 1, Termination, gcomm);
                terminated = true;
                break;
            }
            if (val == Black) {
                if (rank == 0) {
                    assert(color == White);
                } else {
                    color = Black;
                }
            }
            MPI_Send(&color, 1, MPI_INT, ring_next, Token, gcomm);
            color = White;
        }
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
