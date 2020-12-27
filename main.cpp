#include "util/util.h"
#include "util/DeviceMemory.h"
#include "util/Timer.h"

#include "graph_builder/GraphBuilder.h"
#include "graph_builder/Graph.h"

#include <CL/opencl.h>

#include <algorithm>
#include <getopt.h>
#include <iostream>
#include <math.h>


int main(int argc, char **argv) {
  cl_platform_id cpPlatform; // OpenCL platform 
  cl_device_id device_id;    // device ID 
  cl_context context;        // context 
  cl_command_queue queue;    // command queue 
  cl_program program;        // program 

  cl_kernel init_vertex_data;          // kernel
  cl_kernel init_edge_weight;          // kernel
  cl_kernel visit_edges;
  cl_kernel merge;
  cl_kernel find_prev_best_modularity;


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

  // ----- OpenCL initialization begin -----
  //@@ Bind to platform
  clGetPlatformIDs(1, &cpPlatform, NULL);

  //@@ Get ID for the device
  clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  //@@ Create a context
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);

  //@@ Create a command queue
  queue = clCreateCommandQueue(context, device_id, 0, NULL);

  // Read kernel files
  const char* kernel_source_str = read_kernel_from_file("kernel/kernel.cl");

  //@@ Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source_str, NULL, NULL);

  //@@ Build the program executable
  cl_int success = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (success != CL_SUCCESS) {
    printf("kernel build error\n");
    debugKernelSource(program, device_id);
    exit(-1);
  }

  //@@ Create the compute kernel in the program we wish to run
  init_vertex_data = clCreateKernel(program, "init_vertex_data", NULL);
  init_edge_weight = clCreateKernel(program, "init_edge_weight", NULL);
  visit_edges = clCreateKernel(program, "visitEdges", NULL);
  merge = clCreateKernel(program, "merge", NULL);
  find_prev_best_modularity = clCreateKernel(program, "findPrevBestModularity", NULL);


  // ----- OpenCl initialization end -----

  //--- Build graph ---
  Graph* g = (new GraphBuilder(inputFile))->buildGraph();
  g->print();
  unsigned numOfNodes = g->getNumOfNodes();
  unsigned numOfEdges = g->getNumOfEdges();
  unsigned numOfVirtualNodes = g->getNumOfVirtualNodes();
  float totalWeight = g->getTotalWeight();

  // ----- Allocate device memory -----
  // per vertex data
  DeviceMemory<unsigned> d_root(&context, &queue, numOfNodes);
  DeviceMemory<bool> d_is_merged(&context, &queue, numOfNodes);
  DeviceMemory<int> d_weighted_degree(&context, &queue, numOfNodes);
  DeviceMemory<int> d_out_degrees(&context, &queue, numOfNodes);

  // per edge data
  DeviceMemory<unsigned> d_edge_weight(&context, &queue, numOfEdges);

  // graph-related data
  DeviceMemory<unsigned> d_vid_list(&context, &queue, numOfVirtualNodes);
  DeviceMemory<unsigned> d_edge_list(&context, &queue, numOfEdges);

  // ----- Copy from host to device -----
  d_vid_list = g->getVertexIDList();
  d_edge_list = g->getEdgeList();
  d_weighted_degree = g->getOutDegrees();
  d_out_degrees = g->getOutDegrees();

  clFinish(queue);

  // ----- init vertex data -----
  clSetKernelArg(init_vertex_data, 0, sizeof(cl_mem), d_root.getDeviceMemory());
  clSetKernelArg(init_vertex_data, 1, sizeof(cl_mem), d_is_merged.getDeviceMemory());
  clSetKernelArg(init_vertex_data, 2, sizeof(int), &numOfNodes);
  size_t numThreads = 512;
  size_t numGroups = numOfNodes / numThreads + 1;
  size_t numGlobalThreads = numGroups * numThreads;
  success = clEnqueueNDRangeKernel(queue, init_vertex_data, 1, NULL, &numGlobalThreads, &numThreads, 0, NULL, NULL);
  clFinish(queue);
  if (success != CL_SUCCESS) {
    printf("init vertex error\n");
    debug(success);
    exit(-1);
  }

  // ----- init edge weight -----
  clSetKernelArg(init_edge_weight, 0, sizeof(cl_mem), d_edge_weight.getDeviceMemory());
  clSetKernelArg(init_edge_weight, 1, sizeof(int), &numOfEdges);
  numThreads = 512;
  numGroups = numOfNodes / numThreads + 1;
  numGlobalThreads = numGroups * numThreads;
  success = clEnqueueNDRangeKernel(queue, init_edge_weight, 1, NULL, &numGlobalThreads, &numThreads, 0, NULL, NULL);
  clFinish(queue);
  if (success != CL_SUCCESS) {
    printf("init edge weight error\n");
    debug(success);
    exit(-1);
  }

  // ----- Build dendrogram -----
  // argument arrays used for each kernel
  size_t d_numThreads[6];
  size_t d_numGlobalThreads[6];
  int degree[6];
  int logDegree[6];
  unsigned globalNodeOffset[6];
  unsigned globalEdgeOffset[6];
  unsigned totalNumOfNodes[6];
  for (int i = 0; i < 6; ++i) {
    d_numThreads[i] = 256;
    size_t numGroups = (g->getNumOfVirtualNodesAt(i) % 256 == 0) ?
      g->getNumOfVirtualNodesAt(i) / 256 : g->getNumOfVirtualNodesAt(i) / 256 + 1;
    d_numGlobalThreads[i] = d_numThreads[i] * numGroups;
    degree[i] = POW(2, i);
    logDegree[i] = i;
    totalNumOfNodes[i] = g->getNumOfVirtualNodesAt(i);
  }
  unsigned nodeOffset = 0;
  unsigned edgeOffset = 0;
  for (int i = 0; i < 6; ++i) {
    globalNodeOffset[i] = nodeOffset;
    globalEdgeOffset[i] = edgeOffset;
    nodeOffset += g->getNumOfVirtualNodesAt(i);
    edgeOffset += (g->getNumOfVirtualNodesAt(i) * degree[i]);
  }

  Timer t;
  t.start();
  for (int i = 5; i >= 0; --i) {
    numThreads = 512;
    numGroups = numOfNodes / numThreads + 1;
    numGlobalThreads = numGroups * numThreads;

    // merge sources whose degree is 2^i
    clSetKernelArg(merge, 0, sizeof(cl_mem), d_vid_list.getDeviceMemory());
    clSetKernelArg(merge, 1, sizeof(cl_mem), d_weighted_degree.getDeviceMemory());
    clSetKernelArg(merge, 2, sizeof(cl_mem), d_edge_list.getDeviceMemory());
    clSetKernelArg(merge, 3, sizeof(cl_mem), d_edge_weight.getDeviceMemory());
    clSetKernelArg(merge, 4, sizeof(cl_mem), d_root.getDeviceMemory());
    clSetKernelArg(merge, 5, sizeof(cl_mem), d_out_degrees.getDeviceMemory());
    clSetKernelArg(merge, 6, sizeof(cl_mem), d_is_merged.getDeviceMemory());
    clSetKernelArg(merge, 7, sizeof(float), &totalWeight);
    clSetKernelArg(merge, 8, sizeof(unsigned), &totalNumOfNodes[0]);
    clSetKernelArg(merge, 9, sizeof(int), &degree[0]);
    clSetKernelArg(merge, 10, sizeof(int), &logDegree[0]);
    clSetKernelArg(merge, 11, sizeof(unsigned), &globalNodeOffset[0]);
    clSetKernelArg(merge, 12, sizeof(unsigned), &globalEdgeOffset[0]);
    success = clEnqueueNDRangeKernel(queue, merge, 1, NULL, &d_numGlobalThreads[0], &d_numThreads[0], 0, NULL, NULL);
    if (success != CL_SUCCESS) {
      printf("merge function error\n");
      debug(success);
      exit(-1);
    }
    clFinish(queue);
  }
  t.stop();
  std::cout << "Execution time: " << t.elapsedMilliseconds() << " ms" << std::endl;


  // release OpenCL resources 
  clReleaseProgram(program);
  clReleaseKernel(init_vertex_data);
  clReleaseKernel(init_edge_weight);
  clReleaseKernel(merge);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // release host memory

  return 0;
}

