#pragma once

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
    namespace a2 {

        /**
         * A DenseGraph is optimised for a graph in which the number of edges
         * is close to n(n-1). It is represented using an adjacency matrix.
         */
        struct DenseGraph
        {
            std::size_t n; /**< Number of nodes in the graph. */
            node_t* adjacencyMatrix; /** Pointer to an n x n adj. matrix */

            /** Returns number of cells in the adjacency matrix. */
            __device__ __host__ __forceinline__
                std::size_t matrix_size() const { return n * n; }
        };


        namespace gpu {


            /**
             * Constructs a DenseGraph from an input edge list of m edges.
             *
             * @pre The pointers in DenseGraph g have already been allocated.
             */
            __global__
                void build_graph(DenseGraph g, edge_t const* edge_list, std::size_t m)
            {
                // IMPLEMENT ME!

                const int th_id = blockIdx.x * blockDim.x + threadIdx.x;

                if (th_id < m) {

                    edge_t edge = edge_list[th_id];

                    int index = edge.x * g.n + edge.y;

                    g.adjacencyMatrix[index] = 1;

                }

    
                return;
            }

            /**
              * Repopulates the adjacency matrix as a new graph that represents
              * the two-hop neighbourhood of input graph g
              */
            __global__
                void two_hop_reachability(DenseGraph g)
            {
                // IMPLEMENT ME!
                // square adjacencyMatrix
                // then remove the diagonal and clamp values back to [0,1]


                // First approach, use 2D 32x32 threads to precompute row and column number for each thread. Then each thread will represent output matrix by adding along 
                // the row and column of adjacencyMatrix.

                int r = blockIdx.y * blockDim.y + threadIdx.y;

                int c = blockIdx.x * blockDim.x + threadIdx.x;

                int sum = 0;

                if ((r < g.n) && (c < g.n)) {

                    for (int i = 0; i < g.n; i++) {

                        int r1 = g.adjacencyMatrix[r * g.n + i];
                        int r2 = g.adjacencyMatrix[i * g.n + c];

                        sum += g.adjacencyMatrix[r * g.n + i] * g.adjacencyMatrix[i * g.n + c];

                    }

                    g.adjacencyMatrix[r * g.n + c] = sum;
                }

                // post processing


                //if (r == c) {

                //    g.adjacencyMatrix[r * g.n + c] = 0;
                //}

                //else {

                //    if (g.adjacencyMatrix[r * g.n + c] > 0) {
                //        g.adjacencyMatrix[r * g.n + c] = 1;
                //    }
                //}



                return;
            }

            __device__
                void m_square(DenseGraph g) {

            }

        } // namespace gpu
    } // namespace a2
} // namespace csc485b