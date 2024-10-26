#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <iterator> // std::ostream_iterator
#include <vector>
#include <cuda_runtime.h>

#include "device_launch_parameters.h"

#include "dense_graph.h"
#include "sparse_graph.h"

#include "data_generator.h"
#include "data_types.h"

void cpu_dense_two_hop(csc485b::a2::node_t* original_matrix, csc485b::a2::node_t* gpu_result, std::size_t n) {

    // compute matrix squaring on cpu

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            int tmp = 0;

            for (int k = 0; k < n; k++) {

                tmp += original_matrix[i * n + k] * original_matrix[k * n + j];
            }

            // Check against the GPU result

            if (i == j) {
                tmp = 0;
            }

            else {
                if (tmp > 0) {
                    tmp = 1;
                }
            }

            assert(tmp == gpu_result[i * n + j]);
        }
    }
}

void cpu_dense_build(csc485b::a2::node_t* matrix, csc485b::a2::edge_list_t edge_list, std::size_t n, std::size_t m) {

    for (int i = 0; i < m; i++) {
        csc485b::a2::edge_t edge = edge_list[i];

        int index = edge.x * n + edge.y;

        matrix[index] = 1;
        
    }


}

void cpu_sparse_build(csc485b::a2::node_t* neighbours, csc485b::a2::node_t* neighbours_start, csc485b::a2::edge_list_t edge_list, std::size_t n, std::size_t m) {

    for (int i = 0; i < m; i++) {
        int vertex = edge_list[i].x;
        neighbours_start[vertex] += 1;
    }

    for (int i = 1; i < n; i++) {

        neighbours_start[i] = neighbours_start[i] + neighbours_start[i - 1];

    }

    for (int i = n-1; i > 0; i--) {
        neighbours_start[i] = neighbours_start[i - 1];
    }

    neighbours_start[0] = 0;

    for (int i = 0; i < m; i++) {
        int x = edge_list[i].x;
        int y = edge_list[i].y;

        int id = neighbours_start[x];
        neighbours[id] = y;
        neighbours_start[x] += 1;
    }


    for (int i = n - 1; i > 0; i--) {
        neighbours_start[i] = neighbours_start[i - 1];
    }

    neighbours_start[0] = 0;


}

void verify_cpu_sparse(csc485b::a2::node_t* neighbours_original_matrix, csc485b::a2::node_t* neighbours_gpu_result, csc485b::a2::node_t* start_matrix, csc485b::a2::node_t* start_gpu_result, std::size_t n, std::size_t m) {

    for (int i = 0; i < m; i++) {

        assert(neighbours_original_matrix[i] == neighbours_gpu_result[i]);
        
    }

    for (int i = 0; i < n; i++) {

        assert(start_matrix[i] == start_gpu_result[i]);

    }
     
}

/**
 * Runs timing tests on a CUDA graph implementation.
 * Consists of independently constructing the graph and then
 * modifying it to its two-hop neighbourhood.
 */
template < typename DeviceGraph >
void run(DeviceGraph g, csc485b::a2::edge_t const* d_edges, std::size_t m)
{

    unsigned int warp_size = 32;
    unsigned int products_per_block = 1024 / warp_size;
    /*unsigned int num_blocks = std::ceil(g.n * g.n / (double)products_per_block);*/

    /*unsigned int num_blocks = std::ceil(g.n / (double) 32);*/

    unsigned int num_blocks = (g.n + 32 - 1) / 32;


    cudaDeviceSynchronize();
    auto const build_start = std::chrono::high_resolution_clock::now();

    // this code doesn't work yet!
    csc485b::a2::gpu::build_graph << < dim3{num_blocks, num_blocks}, dim3{warp_size, products_per_block} >> > (g, d_edges, m);

    cudaDeviceSynchronize();
    auto const reachability_start = std::chrono::high_resolution_clock::now();

    // neither does this!
    csc485b::a2::gpu::two_hop_reachability << < dim3{num_blocks, num_blocks}, dim3{ warp_size, products_per_block } >> > (g);

    cudaDeviceSynchronize();
    auto const end = std::chrono::high_resolution_clock::now();

    std::cout << "Build time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(reachability_start - build_start).count()
        << " us"
        << std::endl;

    std::cout << "Reachability time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end - reachability_start).count()
        << " us"
        << std::endl;
}

/**
 * Allocates space for a dense graph and then runs the test code on it.
 */
void run_dense(csc485b::a2::edge_t const* d_edges, std::size_t n, std::size_t m, csc485b::a2::edge_list_t edges_cpu)
{
    using namespace csc485b;

    // allocate device DenseGraph
    a2::node_t* d_matrix;
    cudaMalloc((void**)&d_matrix, sizeof(a2::node_t) * n * n);
    cudaMemset(d_matrix, 0, sizeof(a2::node_t) * n * n);
    a2::DenseGraph d_dg{ n, d_matrix };



    std::vector< a2::node_t > initial_host_matrix(d_dg.matrix_size());
    cudaMemcpy(initial_host_matrix.data(), d_matrix, sizeof(a2::node_t) * d_dg.matrix_size(), cudaMemcpyDeviceToHost);
    cpu_dense_build(initial_host_matrix.data(), edges_cpu, n, m);

    run(d_dg, d_edges, m);

    // check output?
    std::vector< a2::node_t > host_matrix(d_dg.matrix_size());
    a2::DenseGraph dg{ n, host_matrix.data() };
    cudaMemcpy(dg.adjacencyMatrix, d_dg.adjacencyMatrix, sizeof(a2::node_t) * d_dg.matrix_size(), cudaMemcpyDeviceToHost);
    /*std::copy(host_matrix.cbegin(), host_matrix.cend(), std::ostream_iterator< a2::node_t >(std::cout, " "));*/

    // verify result

    cpu_dense_two_hop(initial_host_matrix.data(), host_matrix.data(), n);

    // clean up
    cudaFree(d_matrix);
}

/**
 * Allocates space for a sparse graph and then runs the test code on it.
 */
void run_sparse(csc485b::a2::edge_t const* d_edges, std::size_t n, std::size_t m, csc485b::a2::edge_list_t edges_cpu)
{
    using namespace csc485b;

    // allocate device SparseGraph
    a2::node_t* d_offsets, * d_neighbours;
    cudaMalloc((void**)&d_offsets, sizeof(a2::node_t) * n);
    cudaMalloc((void**)&d_neighbours, sizeof(a2::node_t) * m);
    cudaMemset(d_offsets, 0, sizeof(a2::node_t) * n);

    cudaMemset(d_neighbours, 0, sizeof(a2::node_t) * m);
    a2::SparseGraph d_sg{ n, m, d_offsets, d_neighbours };

    std::vector< a2::node_t > initial_host_neighbours_start_at(d_sg.neighbours_start_at_size());
    std::vector< a2::node_t > initial_host_neighbours(d_sg.neighbours_size());

   
    cudaMemcpy(initial_host_neighbours_start_at.data(), d_offsets, sizeof(a2::node_t) * d_sg.neighbours_start_at_size(), cudaMemcpyDeviceToHost);

    
    cudaMemcpy(initial_host_neighbours.data(), d_neighbours, sizeof(a2::node_t) * d_sg.neighbours_size(), cudaMemcpyDeviceToHost);

    cpu_sparse_build(initial_host_neighbours.data(), initial_host_neighbours_start_at.data(), edges_cpu, n, m);

    std::cout << "cpu_start_at: " << std::endl;
    std::copy(initial_host_neighbours_start_at.cbegin(), initial_host_neighbours_start_at.cend(), std::ostream_iterator< a2::node_t >(std::cout, " "));

    std::cout << "cpu_neighbours: " << std::endl;
    std::copy(initial_host_neighbours.cbegin(), initial_host_neighbours.cend(), std::ostream_iterator< a2::node_t >(std::cout, " "));

    run(d_sg, d_edges, m);

    std::cout << "neighbours_start_at: " << std::endl;

    // check output?
    std::vector< a2::node_t > host_neighbours_start_at(d_sg.neighbours_start_at_size());
    std::vector< a2::node_t > host_neighbours(d_sg.neighbours_size());
    a2::SparseGraph sg{ n, m, host_neighbours_start_at.data(), host_neighbours.data()};
    cudaMemcpy(sg.neighbours_start_at, d_sg.neighbours_start_at, sizeof(a2::node_t) * d_sg.neighbours_start_at_size(), cudaMemcpyDeviceToHost);
    std::copy(host_neighbours_start_at.cbegin(), host_neighbours_start_at.cend(), std::ostream_iterator< a2::node_t >(std::cout, " "));

    std::cout << "neighbours: " << std::endl;

    cudaMemcpy(sg.neighbours, d_sg.neighbours, sizeof(a2::node_t) * d_sg.neighbours_size(), cudaMemcpyDeviceToHost);
    std::copy(host_neighbours.cbegin(), host_neighbours.cend(), std::ostream_iterator< a2::node_t >(std::cout, " "));

    verify_cpu_sparse(initial_host_neighbours.data(), host_neighbours.data(), initial_host_neighbours_start_at.data(), host_neighbours_start_at.data(), n, m);

    // clean up
    cudaFree(d_neighbours);
    cudaFree(d_offsets);
}

int main()
{
    using namespace csc485b;

    // Create input
    std::size_t constexpr n = 64;
    std::size_t constexpr expected_degree = n >> 1;

    a2::edge_list_t const graph = a2::generate_graph(n, n * expected_degree);
    std::size_t const m = graph.size();

    /*lazily echo out input graph*/
    //for (auto const& e : graph)
    //{
    //    std::cout << "(" << e.x << "," << e.y << ") ";
    //}


    // allocate and memcpy input to device
    a2::edge_t* d_edges;
    cudaMalloc((void**)&d_edges, sizeof(a2::edge_t) * m);
    cudaMemcpyAsync(d_edges, graph.data(), sizeof(a2::edge_t) * m, cudaMemcpyHostToDevice);

    // run your code!
    /*run_dense(d_edges, n, m, graph);*/
    run_sparse(d_edges, n, m, graph);

    return EXIT_SUCCESS;
}