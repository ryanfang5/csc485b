#pragma once

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
    namespace a2 {

        /**
         * A SparseGraph is optimised for a graph in which the number of edges
         * is close to cn, for a small constanct c. It is represented in CSR format.
         */
        struct SparseGraph
        {
            std::size_t n; /**< Number of nodes in the graph. */
            std::size_t m; /**< Number of edges in the graph. */
            node_t* neighbours_start_at; /** Pointer to an n=|V| offset array */
            node_t* neighbours; /** Pointer to an m=|E| array of edge destinations */

            __device__ __host__ __forceinline__
                std::size_t neighbours_start_at_size() const { return n; }

            __device__ __host__ __forceinline__
                std::size_t neighbours_size() const { return m; }
        };


        namespace gpu {

            /**
             * Constructs a SparseGraph from an input edge list of m edges.
             *
             * @pre The pointers in SparseGraph g have already been allocated.
             */
            __global__
                void build_graph(SparseGraph g, edge_t const* edge_list, std::size_t m)
            {
                // IMPLEMENT ME!
                int r = blockIdx.y * blockDim.y + threadIdx.y;

                int c = blockIdx.x * blockDim.x + threadIdx.x;

                const int th_id = r * blockDim.y + c;

                // CPU version


                //if (th_id == 0) {

                //    for (int i = 0; i < m; i++) {
                //        int vertex = edge_list[i].x;
                //        g.neighbours_start_at[vertex] += 1;
                //    }

                //    for (int i = 1; i < g.n; i++) {

                //        g.neighbours_start_at[i] += g.neighbours_start_at[i - 1];

                //    }

                //    for (int i = g.n - 1; i > 0; i--) {
                //        g.neighbours_start_at[i] = g.neighbours_start_at[i - 1];
                //    }

                //    g.neighbours_start_at[0] = 0;

                //    for (int i = 0; i < m; i++) {
                //        int x = edge_list[i].x;
                //        int y = edge_list[i].y;

                //        int id = g.neighbours_start_at[x];
                //        g.neighbours[id] = y;
                //        g.neighbours_start_at[x] += 1;
                //    }


                //    for (int i = g.n - 1; i > 0; i--) {
                //        g.neighbours_start_at[i] = g.neighbours_start_at[i - 1];
                //    }

                //    g.neighbours_start_at[0] = 0;
                //}




                // GPU version


                if (th_id < m) {
                    int vertex = edge_list[th_id].x;
                    atomicAdd(g.neighbours_start_at+vertex, 1);
                }

                for (int i = 1; i < g.n; i <<= 1) {

                    if (th_id >= i && th_id < g.n) {
                        g.neighbours_start_at[th_id] = g.neighbours_start_at[th_id] + g.neighbours_start_at[th_id - i];
                    }

                    __syncthreads();
                }

                if (th_id < g.n) {

                    int val = g.neighbours_start_at[th_id];

                    g.neighbours_start_at[th_id] = __shfl_up_sync(0xFFFFFFFF, val, 1);

                }


                if (th_id == 0) {

                    g.neighbours_start_at[0] = 0;

                    for (int i = 0; i < m; i++) {
                        int x = edge_list[i].x;
                        int y = edge_list[i].y;

                        int id = g.neighbours_start_at[x];
                        g.neighbours[id] = y;
                        g.neighbours_start_at[x] += 1;
                    }
                }


                if (th_id < g.n) {

                    int val = g.neighbours_start_at[th_id];

                    g.neighbours_start_at[th_id] = __shfl_up_sync(0xFFFFFFFF, val, 1);

                }

                if (th_id == 0) {
                    g.neighbours_start_at[0] = 0;
                }




                return;
            }

            /**
              * Repopulates the adjacency lists as a new graph that represents
              * the two-hop neighbourhood of input graph g
              */
            __global__
                void two_hop_reachability(SparseGraph g)
            {
                // IMPLEMENT ME!
                // algorithm unknown
                return;
            }

        } // namespace gpu
    } // namespace a2
} // namespace csc485b