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

#include "pr.cuh"

#define ITERATION (20)

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

  // ----- DeviceMemory initialization for graph representation -----
  DeviceMemory<uint32_t> d_vid_list(numOfVirtualNodes);
  DeviceMemory<uint32_t> d_edge_list(numOfEdges);

  // ----- DeviceMemory initialization for application-specific data -----
  DeviceMemory<float> d_contrib(numOfNodes);
  DeviceMemory<float> d_incoming_total(numOfNodes);
  DeviceMemory<int> d_out_degrees(numOfNodes);

  // Host Side
  int* h_outDegrees = g->getOutDegrees();
  float* h_incomingTotal = new float[numOfNodes];
  float* h_contrib = new float[numOfNodes];

  #pragma omp parallel for
  for (uint32_t i = 0; i < numOfNodes; ++i) {
    h_incomingTotal[i] = 0;
    h_contrib[i] = (1.0f / numOfNodes) / h_outDegrees[i];
  }

  // memcopy from host to device
  d_vid_list = g->getVertexIDList();
  d_edge_list = g->getEdgeList();

  d_contrib = h_contrib;
  d_incoming_total = h_incomingTotal;
  d_out_degrees = g->getOutDegrees();

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

  Timer t;
  t.start();
  for (int iter = 0; iter < ITERATION; ++iter) {
    for (uint32_t i = 0; i < 6; ++i) {
      uint32_t numOfCurrentVirtualNodes = g->getNumOfVirtualNodesAt(i);
      uint32_t numBlocks = (numOfCurrentVirtualNodes % 512 == 0) ? 
                           numOfCurrentVirtualNodes / 512:
                           numOfCurrentVirtualNodes / 256 + 1;
      if (numBlocks != 0) {
        pr<<<numBlocks, 512, 0, st[i]>>>(
            d_incoming_total.getDevicePtr(),
            d_contrib.getDevicePtr(),
            d_vid_list.getDevicePtr() + vertex_offset[i],
            d_edge_list.getDevicePtr() + edge_offset[i],
            degree_list[i],
            i,
            numOfCurrentVirtualNodes);
      }
    }
    cudaErrorCheck(cudaDeviceSynchronize());
    pr_update<<<numOfNodes / 1024 + 1, 1024>>>(
        d_incoming_total.getDevicePtr(),
        d_contrib.getDevicePtr(),
        d_out_degrees.getDevicePtr(),
        (1.0f - 0.85) / numOfNodes,
        numOfNodes);
    cudaErrorCheck(cudaDeviceSynchronize());
  }
  t.stop();

  return 0;
}


