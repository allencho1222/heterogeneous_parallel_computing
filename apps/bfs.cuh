#include <cuda.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

#define BFS_MAX (1073741824)

typedef cub::BlockScan<uint32_t, 256> BlockScan;

__global__
void bfs(bool* d_finished,
         uint32_t* d_distance,
         uint32_t level,
         uint32_t* d_vertexIDList,
         uint32_t* d_edgeList,
         uint32_t degree,
         uint32_t logDegree,
         uint32_t totalNumOfNodes) {
  __shared__ typename BlockScan::TempStorage temp_storage; 
  __shared__ uint32_t offsetList[256];

  uint32_t nodeOffset = blockIdx.x << 8;  // blockIdx.x * 256
  uint32_t edgeOffset = nodeOffset << logDegree; // nodeOffset * degree
  uint32_t numOfNodes = (nodeOffset + 256 > totalNumOfNodes) ?
                        totalNumOfNodes - nodeOffset : 256;

  uint32_t tid = threadIdx.x;
  uint32_t sourceID;
  uint32_t need_to_process = 0;
  uint32_t offset= 0;
  uint32_t numOfActiveThreads = 0;

  if (tid < numOfNodes) {
    sourceID = d_vertexIDList[nodeOffset + tid];
    need_to_process = (cub::ThreadLoad<cub::LOAD_CA>(d_distance + sourceID) == level);
  }
  BlockScan(temp_storage).ExclusiveSum(need_to_process, offset, numOfActiveThreads);
  if (numOfActiveThreads == 0)
	  return;

  if (need_to_process) {
    offsetList[offset] = tid << logDegree;
  }
  __syncthreads();

  uint32_t laneID = tid & (degree - 1);
  uint32_t shmIndex = tid >> logDegree;
  while (shmIndex < numOfActiveThreads) {
    uint32_t edgeIndex = edgeOffset + offsetList[shmIndex] + laneID;
    uint32_t destID = d_edgeList[edgeIndex];

    if (d_distance[destID] == BFS_MAX) {
      d_distance[destID] = level + 1;
      *d_finished = false;
    }

    tid += blockDim.x;
    shmIndex = tid >> logDegree;
  }
}
