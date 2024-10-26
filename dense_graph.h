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

                int r = blockIdx.y * blockDim.y + threadIdx.y;

                int c = blockIdx.x * blockDim.x + threadIdx.x;

                const int th_id = r * blockDim.y + c;

                if (th_id < m) {

                    edge_t edge = edge_list[th_id];

                    int index = edge.x * g.n + edge.y;

                    g.adjacencyMatrix[index] = 1;

                }

                return;
            }

            __device__
                void tiled_two_hop(DenseGraph g, int* smema, int* smemb, int index, int r, int c) {

                //__shared__ int smema[1024];
                //__shared__ int smemb[1024];

                //int r = blockIdx.y * blockDim.y + threadIdx.y;
                //int c = blockIdx.x * blockDim.x + threadIdx.x;

                int sum = 0;

                /*int index = threadIdx.y * blockDim.x + threadIdx.x;*/ // map thread to 1D index for smem. Since we are deploying 1024 threads in a block, this will range from 0-1023

                if ((r < g.n) && (c < g.n)) {

                    // using tile size of blockDim (32 x 32)

                    for (int i = 0; i < g.n; i += blockDim.x) {

                        // Load corresponding elements for each tile

                        smema[index] = g.adjacencyMatrix[(r * g.n) + (i + threadIdx.x)]; 
                        // Same as naive implementation, except the column will change by adding the offset (i) to locate the correct block, along with the local thread id.x for the exact column.

                        smemb[index] = g.adjacencyMatrix[(i + threadIdx.y) * g.n + c];
                        // Add the offset (i) along with local thread id.y for the correct row, then add global column.


                        // Wait for tiles to be loaded 
                        __syncthreads();

                        // Perform partial dot product
                        for (int j = 0; j < blockDim.x; j++) {
                            sum += smema[threadIdx.y * blockDim.x + j] * smemb[j * blockDim.x + threadIdx.x];
                        }

                        // Wait for threads to finish adding before loading new ones
                        __syncthreads();
                    }

                    // Clamp values to [0, 1] and write result to global memory

                    if (r == c) {
                        g.adjacencyMatrix[r * g.n + c] = 0;
                    }

                    else {
                        if (sum > 0) {
                            g.adjacencyMatrix[r * g.n + c] = 1;
                        }
                    }

                    /*g.adjacencyMatrix[r * g.n + c] = sum;*/

                }

                return;

            }

            __device__
                void naive_two_hop(DenseGraph g) {

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

                __shared__ int smema[1024];
                __shared__ int smemb[1024];

                int r = blockIdx.y * blockDim.y + threadIdx.y;
                int c = blockIdx.x * blockDim.x + threadIdx.x;

                int sum = 0;

                int index = threadIdx.y * blockDim.x + threadIdx.x;

                if (r < g.n && c < g.n) {
                    tiled_two_hop(g, smema, smemb, index, r, c);

                    // naive_two_hop(g);
                }




                return;
            }

         

        } // namespace gpu
    } // namespace a2
} // namespace csc485b