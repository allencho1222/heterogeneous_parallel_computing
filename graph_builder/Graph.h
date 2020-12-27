#ifndef GRAPH_H
#define GRAPH_H

#include <assert.h>
#include <math.h>
#include <algorithm>

#define POW(a, b) ((unsigned)pow(a, b))

class Graph {
public:
  Graph(unsigned numOfNodes, unsigned numOfEdges, unsigned* numOfVirtualNodes,
        unsigned* outDegrees)
  : numOfNodes(numOfNodes), numOfEdges(numOfEdges) {
    // the size of numOfVirtualNodes must be 6
    for (int i = 0; i < 6; ++i) {
      this->numOfVirtualNodes[i] = numOfVirtualNodes[i];
      numOfTotalVirtualNodes += numOfVirtualNodes[i];
    }

    vertexIDList = new unsigned[numOfTotalVirtualNodes];
    edgeList = new unsigned[numOfEdges];

    this->outDegrees = new int[numOfNodes];
    for (unsigned i = 0; i < numOfNodes; ++i) {
      this->outDegrees[i] = (int)outDegrees[i];
      totalWeight += outDegrees[i];
    }
  }
  ~Graph() {
    delete vertexIDList;
    delete edgeList;
    printf("Free vertexIDList\n");
    printf("Free edgeList\n");
  }

  unsigned getNumOfNodes() { return numOfNodes; }
  unsigned getNumOfEdges() { return numOfEdges; }
  float getTotalWeight() { return totalWeight; }

  unsigned* getVertexIDList() { return vertexIDList; }
  unsigned* getEdgeList() { return edgeList; }
  int* getOutDegrees() { return outDegrees; }

  unsigned getNumOfVirtualNodes() { return numOfTotalVirtualNodes; }
  unsigned getNumOfVirtualNodesAt(unsigned logDegree) { 
    return numOfVirtualNodes[logDegree]; 
  }

  void print() {
    unsigned totalNumOfEdges = 0;
    printf("##### Print regularized graph #####\n");
    printf("# of nodes: %lu\n", numOfNodes);
    printf("# of edges: %lu\n", numOfEdges);
    for (int i = 0; i < 6; ++i) {
      printf("Degree: %d\n", POW(2, i));
      printf("\t# of virtual nodes: %lu\n", getNumOfVirtualNodesAt(i));
      printf("\t# of edges: %lu\n", getNumOfVirtualNodesAt(i) * POW(2, i));
      totalNumOfEdges += getNumOfVirtualNodesAt(i) * POW(2, i);
    }
    assert(totalNumOfEdges == numOfEdges);

    printf("First 20 nodes\n");
    int iter = std::min(20, (int)numOfNodes);
    for (int i = 0; i < iter; ++i) {
      printf("vertex %d, degree: %d\n", i, outDegrees[i]);
    }
  }

private:
  unsigned numOfNodes;
  unsigned numOfEdges;
  unsigned numOfTotalVirtualNodes;
  unsigned numOfVirtualNodes[6];

  float totalWeight;

  unsigned* vertexIDList;
  unsigned* edgeList;
  int* outDegrees;
};

#endif


