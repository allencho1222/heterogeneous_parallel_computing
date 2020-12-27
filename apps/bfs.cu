#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <getopt.h>

#include "../graph_builder/GraphBuilder.h"
#include "../graph_builder/Graph.h"

#include "../util/Timer.h"
#include "../util/CudaHelper.h"
#include "../util/DeviceMemoryCUDA.h"

#include "bfs.cuh"

int main(int argc, char** argv) {
  std::string inputFile;
  // ----- Parse command line argument -----
  extern char* optarg;
  char c;
  while ((c = getopt(argc, argv, "f:")) != -1) {
    switch (c) {
      case 'f':
        inputFile = std::string(optarg);
        break;
      default:
        printf("wrong argument\n");
        exit(-1);
    }
  }

  //--- Build graph ---
  Graph* g = (new GraphBuilder(inputFile))->buildGraph();
  g->print();
  unsigned numOfNodes = g->getNumOfNodes();
  unsigned numOfEdges = g->getNumOfEdges();
  unsigned numOfVirtualNodes = g->getNumOfVirtualNodes();

  // ----- Allocate device memory -----
  DeviceMemory<unsigned> d_vid_list(numOfVirtualNodes);
  DeviceMemory<unsigned> d_edge_list(numOfEdges);
  DeviceMemory<unsigned> d_out_degrees(numOfNodes);
  DeviceMemory<unsigned> d_distance(numOfNodes);

  uint32_t* h_distance = new uint32_t[numOfNodes];
  #pragma omp parallel for
  for (uint32_t i = 0; i < numOfNodes; ++i) {
    h_distance[i] = BFS_MAX;
  }

  h_distance[0] = 0;

  // memcopy from host to device
  d_vid_list = g->getVertexIDList();
  d_edge_list = g->getEdgeList();

  d_distance = h_distance;

  // ----- run graph application -----
  cudaStream_t st[6];
  for (int i = 0; i < 6; ++i)
    cudaStreamCreate(&st[i]);

  uint32_t degree_list[6] = {1, 2, 4, 8, 16, 32};
  uint32_t vertex_offset[6];
  uint32_t edge_offset[6];
  
  uint32_t vertex_offset_sum = 0;
  uint32_t edge_offset_sum = 0;
  for (uint32_t i = 0; i < 6; ++i) {
    vertex_offset[i] = vertex_offset_sum;
    edge_offset[i] = edge_offset_sum;
    vertex_offset_sum += g->getNumOfVirtualNodesAt(i);
    edge_offset_sum += (g->getNumOfVirtualNodesAt(i) * degree_list[i]);
  }

  bool h_finished;
  bool* d_finished;
  cudaErrorCheck(cudaMalloc((void**)&d_finished, sizeof(bool)));
  uint32_t iter = 0;
  uint32_t level = 0;
  Timer t;
  t.start();
  do {
    iter++;
    h_finished = true;
    cudaErrorCheck(cudaMemcpy(d_finished, &h_finished, sizeof(bool), cudaMemcpyHostToDevice));
    for (uint32_t i = 0; i < 6; ++i) {
      uint32_t numOfCurrentVirtualNodes = g->getNumOfVirtualNodesAt(i);
      uint32_t numBlocks = (numOfCurrentVirtualNodes % 256 == 0) ? 
                           numOfCurrentVirtualNodes / 256 :
                           numOfCurrentVirtualNodes / 256 + 1;
      if (numBlocks != 0) {
        bfs<<<numBlocks, 256, 0, st[i]>>>(
            d_finished,
            d_distance.getDevicePtr(),
            level,
            d_vid_list.getDevicePtr() + vertex_offset[i],
            d_edge_list.getDevicePtr() + edge_offset[i],
            degree_list[i],
            i,
            numOfCurrentVirtualNodes);
      }
    }

    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(&h_finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
    level++;
  } while (!(h_finished));

  t.stop();

  return 0;
}


