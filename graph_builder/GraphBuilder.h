#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include <vector>
#include <fstream>

#include "Graph.h"
#include "platform_atomics.h"

class GraphBuilder {
public:
  // pair<source vertex, destination vertex>
  typedef std::pair<unsigned, unsigned> Edge;

  GraphBuilder(std::string edgeFileName);

  Graph* buildGraph();

private:
  unsigned numOfNodes;
  unsigned numOfEdges;
  std::vector<Edge> builderEdgeList;

  // graph file
  std::ifstream edgeFile;
  std::string edgeFileName;

  // intermediate results to build the graph
  unsigned* outDegrees;

  unsigned numOfVirtualNodes_out[6];
  unsigned numOfTotalVirtualNodes_out;

  unsigned nodePrefixSum_out[6];
  unsigned edgePrefixSum_out[6];
  unsigned** edgePosition_out;
  unsigned** remains_out;



  // Functions to build the graph
  void readEdges();

  unsigned countNumOfNodes();
  void countDegrees();

  void initIntermediateResult();
  void freeIntermediateResult();

  void countNumOfVirtualNodes();
  void calculatePrefixSum();
  void buildVertexIDList(Graph* g);
  void buildEdgeList(Graph* g);
};

#endif
