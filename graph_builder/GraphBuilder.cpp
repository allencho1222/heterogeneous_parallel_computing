#include <iostream>
#include <math.h>
#include <assert.h>

#include "GraphBuilder.h"

#define POW(a, b) ((unsigned)pow(a, b))

GraphBuilder::GraphBuilder(std::string edgeFileName) :
    edgeFile(edgeFileName), edgeFileName(edgeFileName) {
  numOfNodes = 0;
  numOfEdges = 0;
  numOfTotalVirtualNodes_out = 0;
}

// Read edge file
void GraphBuilder::readEdges() {
  unsigned src, dest;
  
  if (!edgeFile.is_open()) {
    std::cout << "file open fail: " << edgeFileName << std::endl;
    std::exit(-1);
  }

  // parse edge list from the file
  while (edgeFile >> src >> dest) {
    if (src != dest) // ignore self-edge
      builderEdgeList.push_back(std::make_pair(src, dest));
  }

  std::cout << "finish reading the file" << std::endl;
  printf("numOfEdges: %lu\n", builderEdgeList.size());
}

Graph* GraphBuilder::buildGraph() {
  readEdges();

  numOfNodes = countNumOfNodes() + 1;
  numOfEdges = builderEdgeList.size();
  printf("numOfNodes: %d\n", numOfNodes);
  initIntermediateResult();

  countDegrees();
  countNumOfVirtualNodes();
  calculatePrefixSum();

  Graph* g = 
    new Graph(numOfNodes, numOfEdges, numOfVirtualNodes_out, outDegrees);
  buildVertexIDList(g);
  buildEdgeList(g);

  freeIntermediateResult();

  std::cout << "Finish building graph" << std::endl;

  return g;

}

unsigned GraphBuilder::countNumOfNodes() {
  unsigned curMax = 0;
  #pragma omp parallel for reduction(max : curMax)
  for (auto it = builderEdgeList.begin(); it < builderEdgeList.end(); ++it) {
    Edge edge = *it;
    curMax = std::max(curMax, edge.first);
    curMax = std::max(curMax, edge.second);
  }
  return curMax;
}

void GraphBuilder::countDegrees() {
  #pragma omp parallel for
  for (auto it = builderEdgeList.begin(); it < builderEdgeList.end(); ++it) {
    Edge edge = *it;
    unsigned src = edge.first;
    unsigned dest = edge.second;

    fetch_and_add(outDegrees[src], 1);
  }
}

void GraphBuilder::initIntermediateResult() {
  edgePosition_out = new unsigned*[6];
  remains_out = new unsigned*[6];
  for (unsigned i = 0; i < 6; ++i) {
    edgePosition_out[i] = new unsigned[numOfNodes];
    remains_out[i] = new unsigned[numOfNodes];
  }

  outDegrees = new unsigned[numOfNodes];
  #pragma omp parallel for
  for (unsigned i = 0 ; i < numOfNodes; ++i) {
    outDegrees[i] = 0;
    for (unsigned j = 0; j < 6; ++j) {
      edgePosition_out[j][i] = 0;
      remains_out[j][i] = 0;
    }
  }

  for (unsigned i = 0; i < 6; ++i) {
    numOfVirtualNodes_out[i] = 0;
    nodePrefixSum_out[i] = 0;
    edgePrefixSum_out[i] = 0;
  }
}

void GraphBuilder::freeIntermediateResult() {
  for (unsigned i = 0; i < 6; ++i) {
    delete edgePosition_out[i];
    delete remains_out[i];
  }
  delete edgePosition_out;
  delete remains_out;

  delete outDegrees;

  builderEdgeList.clear();
}

void GraphBuilder::countNumOfVirtualNodes() {
  #pragma omp parallel for
  // for each node
  for (unsigned i = 0; i < numOfNodes; ++i) {
    unsigned numOfNodes_i_out[6];
    unsigned numOfRemains_i_out[6];

    // for outgoing edges
    numOfNodes_i_out[5] = outDegrees[i] / 32; // outDegrees[i] / 32
    numOfRemains_i_out[5] = outDegrees[i] % 32; // outDegrees[i] % 32

    for (int j = 4; j >= 0; --j) {
      // for outgoing edges
      numOfNodes_i_out[j] = numOfRemains_i_out[j + 1] / POW(2, j);
      numOfRemains_i_out[j] = numOfRemains_i_out[j + 1] % POW(2, j);
    }

    // after calcuating the number of virtual nodes of 'i' node
    for (unsigned k = 0; k < 6; ++k) {
      // ignore if the number of virtual nodes is zero
      if (numOfNodes_i_out[k] != 0)
        fetch_and_add(numOfVirtualNodes_out[k], numOfNodes_i_out[k]);
    }
  }

  for (unsigned i = 0; i < 6; ++i) {
    numOfTotalVirtualNodes_out += numOfVirtualNodes_out[i];
  }
}

void GraphBuilder::calculatePrefixSum() {
  // prefix sum for nodes
  unsigned nodeCurrent_out = 0;

  // prefix sum for edges
  unsigned edgeCurrent_out = 0;
  for (unsigned i = 0; i < 6; ++i) {
    nodePrefixSum_out[i] = nodeCurrent_out;
    edgePrefixSum_out[i] = edgeCurrent_out;

    nodeCurrent_out += numOfVirtualNodes_out[i];
    edgeCurrent_out += (numOfVirtualNodes_out[i] * POW(2, i));
  }
}

void GraphBuilder::buildVertexIDList(Graph* g) {
  assert(numOfTotalVirtualNodes_out != 0);
  unsigned* vertexIDList = g->getVertexIDList();
  for (unsigned nodeID = 0; nodeID < numOfNodes; ++nodeID) {
    if (outDegrees[nodeID] != 0) {
      unsigned currentRemain = outDegrees[nodeID];
      for (int i = 5; i >= 0; --i) {
        unsigned degrees = POW(2, i);
        if (currentRemain >= degrees) {
          unsigned numOfVirtualNodes = currentRemain >> i;
          unsigned start = nodePrefixSum_out[i];
          unsigned end = start + numOfVirtualNodes;
          for (unsigned pos = start; pos < end; ++pos) {
            vertexIDList[pos] = nodeID;
            //sparse_vertex_thread_map[nodeID] = pos;
          }
          nodePrefixSum_out[i] += numOfVirtualNodes;
          edgePosition_out[i][nodeID] = edgePrefixSum_out[i];
          remains_out[i][nodeID] = (numOfVirtualNodes * degrees);
          edgePrefixSum_out[i] += (degrees * numOfVirtualNodes);

        }
        currentRemain = currentRemain & (degrees - 1);
      }
    }
  }
}

void GraphBuilder::buildEdgeList(Graph* g) {
  unsigned* edgeList = g->getEdgeList();
  for (auto it = builderEdgeList.begin(); it < builderEdgeList.end(); ++it) {
    Edge edge = *it;
    unsigned src = edge.first;
    unsigned dest = edge.second;

    for (int i = 0; i >= 0; --i) {
      if (remains_out[i][src] > 0) {
        edgeList[edgePosition_out[i][src]] = dest;
        edgePosition_out[i][src]++;
        remains_out[i][src]--;
        break;
      }
    }
  }
}
