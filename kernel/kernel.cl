// Initialize vertex data
__kernel
void init_vertex_data(__global unsigned* root,
                      __global bool* isMerged,
                      int numOfNodes) {
  int vid = get_global_id(0);
  if (vid < numOfNodes) {
    root[vid] = vid;
    isMerged[vid] = false;
  }
}

// Initialize edge weight
__kernel
void init_edge_weight(__global unsigned* edgeWeight,
                      unsigned numOfEdges) {
  int eid = get_global_id(0);
  if (eid < numOfEdges) {
    edgeWeight[eid] = 1.0f;
  }
}

// Block-level inclusive scan
unsigned inclusiveScan(unsigned need_to_process, volatile __local unsigned* smem) {
  int tid = get_local_id(0);
  smem[tid] = need_to_process;

  for (unsigned stride = 1; stride <= 256; stride <<= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int idx = (tid + 1) * 2 * stride - 1;
    if (idx < 256) {
      smem[idx] += smem[idx - stride];
    }
  }
  for (unsigned stride = 256 / 2; stride >0; stride >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx + stride < 256) {
      smem[idx + stride] += smem[idx];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  return smem[tid];
}

// find destination ID that gives the highest modularity gain
// Here, I assume that the all threads in a warp execute in lock-step manner
// (e.g., cannot execute different PCs between threads in the warp).
// Warp-level primitives (e.g., warp-level sync) are not supported on OpenCL 1.2

float findBestModularity(int baseThreadID, unsigned *bestThreadID, int degree,
                         unsigned edgeWeight, float totalWeight, 
                         int srcStr, int destStr) {
  int tid = get_local_id(0);
  volatile __local int bestTID[256];
  volatile __local unsigned reduce_smem[256];
  reduce_smem[tid] = (float)edgeWeight - ((float)srcStr * destStr) / totalWeight;
  bestTID[tid] = tid;

  // This is warp-level reduction
  for (int stride = degree / 2; stride > 0; stride >>= 1) {
    if (reduce_smem[tid] < reduce_smem[tid + stride]) {
      reduce_smem[tid] = reduce_smem[tid + stride];
      bestTID[tid] = bestTID[tid + stride];
    }
  }

  *bestThreadID = bestTID[baseThreadID];
  return reduce_smem[baseThreadID];
}

unsigned traceRoot(__global unsigned* root, unsigned destID) {
  unsigned _root = destID;
  for (;;) {
    unsigned __root = root[_root];
    if (__root == _root)
      break;
    _root = __root;
  }


  if (destID != _root && root[destID] != _root) {
    root[destID] = _root;
  }

  return _root;
}

// Merge source vertex into destination vertex
__kernel
void merge(__global unsigned* vertexIDList,
           __global int* str,
           __global unsigned* edgeList,
           __global unsigned* edgeWeight,
           __global unsigned* root,
           __global int* outDegrees,
           __global bool* isMerged,
           float totalWeight,
           unsigned totalNumOfNodes, int degree, int logDegree, 
           unsigned globalNodeOffset, unsigned globalEdgeOffset) {
  volatile __local unsigned scan_smem[256];
  volatile __local unsigned approximatedWeight[256];
  __local unsigned sourceStrList[256];
  __local unsigned sourceIDList[256];
  __local unsigned offsetList[256];

  unsigned nodeOffset = get_group_id(0) << 8;
  unsigned edgeOffset = nodeOffset << logDegree;
  unsigned numOfNodes = (nodeOffset + 256 > totalNumOfNodes) ?
                        totalNumOfNodes - nodeOffset : 256;

  unsigned tid = get_local_id(0);
  unsigned sourceID;
  unsigned need_to_process = 0;

  if (tid < numOfNodes) {
    sourceID = vertexIDList[globalNodeOffset + nodeOffset + tid];
    need_to_process = (unsigned) (!isMerged[sourceID]);
  }

  unsigned offset = inclusiveScan(need_to_process, scan_smem) - 1;
  // There's nothing to schedule
  if (scan_smem[255] == 0)
    return;

  // Coalesce vertices to be scheduled.
  if (need_to_process) {
    offsetList[offset] = tid << logDegree;
    sourceIDList[offset] = sourceID;
  }

  unsigned baseThreadID = (tid >> logDegree) * degree;
  unsigned edgeWay = tid & (degree - 1); // tid % degree
  unsigned shmOffset = tid >> logDegree; // tid / degree
  unsigned iterIdx = tid;
  // Each thread consecutively access 'edgeList' & perfectly balanced
  // (but experience different memory latencies when accessing array by using 'destID')
  // This loop tries to merge source vertex into destination vertex
  while (shmOffset < scan_smem[255]) {
    unsigned edgeIdx = globalEdgeOffset + edgeOffset + offsetList[shmOffset] + edgeWay;
    unsigned destID = edgeList[edgeIdx];
    unsigned srcID = sourceIDList[shmOffset];

    // find the root where the destination is located
    destID = traceRoot(root, destID);

    // warp-level
    approximatedWeight[tid] = 0;

    // approximately calculate new edge weight
    atomic_add(&approximatedWeight[destID % 2 + (baseThreadID << logDegree)], edgeWeight[edgeIdx]);

    // edge and weight are updated
    edgeList[edgeIdx] = destID;
    edgeWeight[edgeIdx] = approximatedWeight[destID % 2 + (baseThreadID << logDegree)];

    if (tid == baseThreadID)
      sourceStrList[baseThreadID] = atomic_xchg(&str[srcID], -1);

    if (tid == baseThreadID)
      sourceStrList[baseThreadID] = str[srcID];

    // findBestModularity() returns the best modularity improvement.
    // In addition, the function saves corresponding thread ID into 'bestThreadID'
    unsigned bestThreadID = baseThreadID;
    float best = 
      findBestModularity(baseThreadID, &bestThreadID,
                         degree, edgeWeight[edgeIdx], totalWeight, 
                         sourceStrList[baseThreadID], str[destID]);

    if (tid == bestThreadID) {
      // No improvement in moduarlity
      if (best < 0.0f) {
        // recover the original value
        atomic_xchg(&str[srcID], sourceStrList[shmOffset]);
      } else {
        float destStr = atomic_add(&str[destID], 0); // atomic load
        // If the destination vertex is not trying to be merged into other vertices
        // by other threads,
        if (destStr >= 0.0) {
          if (atomic_cmpxchg(&str[destID], destStr, sourceStrList[shmOffset] + destStr) == destStr) {
            // srcID is merged into destID
            root[srcID] = destID;
            isMerged[srcID] = true;
          } else {
            atomic_xchg(&str[srcID], sourceStrList[shmOffset]);
          }
        } else {
          atomic_xchg(&str[srcID], sourceStrList[shmOffset]);
        }
      }
    }

    iterIdx += 256;
    shmOffset = iterIdx >> logDegree;
  }
}
