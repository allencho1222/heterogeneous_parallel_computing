#include <cuda.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

__global__
void pr_update(float* d_incomingTotal,
               float* d_contrib,
               int32_t* d_outDegrees,
               float base_score,
               uint32_t numOfNodes) {
  uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < numOfNodes) {
    float score = base_score + 0.85 * d_incomingTotal[tid];
    d_contrib[tid] = score / d_outDegrees[tid];
    d_incomingTotal[tid] = 0;
  }
}


__global__
void pr(float* d_incomingTotal,
        float* d_contrib,
        uint32_t* d_vertexIDList,
        uint32_t* d_edgeList,
        uint32_t degree,
        uint32_t logDegree,
        uint32_t totalNumOfNodes) {
  uint32_t nodeOffset = blockIdx.x << 9;  // blockIdx.x * 512
  uint32_t edgeOffset = nodeOffset << logDegree; // nodeOffset * degree
  uint32_t numOfNodes = (nodeOffset + 512 > totalNumOfNodes) ?
                        totalNumOfNodes - nodeOffset : 512;
  uint32_t tid = threadIdx.x;

  uint32_t laneID = tid & (degree - 1);
  uint32_t shmIndex = tid >> logDegree;
  while (shmIndex < numOfNodes) {
    uint32_t sourceID = d_vertexIDList[nodeOffset + shmIndex];
    uint32_t edgeIndex = edgeOffset + (shmIndex << logDegree) + laneID;
    uint32_t destID = d_edgeList[edgeIndex];

    atomicAdd(&d_incomingTotal[destID], d_contrib[sourceID]);

    tid += blockDim.x;
    shmIndex = tid >> logDegree;
  }
}

