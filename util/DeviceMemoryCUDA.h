#ifndef DEVICE_MEMORY_H
#define DEVICE_MEMORY_H

#include <iostream>
#include <cuda.h>

#include "CudaHelper.h"

template <typename T>
class DeviceMemory {
public:
  DeviceMemory(uint32_t numOfElements) {
    cudaErrorCheck(cudaMalloc((void**)&devicePtr, sizeof(T) * numOfElements));
    this->numOfElements = numOfElements;
  }

  uint64_t getAllocSize() { return sizeof(T) * numOfElements; }

  void operator=(T* hostPtr) {
    cudaErrorCheck(cudaMemcpy(devicePtr, 
                              hostPtr, 
                              sizeof(T) * numOfElements, 
                              cudaMemcpyHostToDevice));
  }

  T* getDevicePtr() { return this->devicePtr; }

  void printResult() {
    T* result = new T[numOfElements];
    cudaErrorCheck(cudaMemcpy(result, devicePtr, 
                              sizeof(T) * numOfElements, 
                              cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < numOfElements; ++i)
      std::cout << i << ":\t" << result[i] << std::endl;
  }


private:
  T* devicePtr;

  uint32_t numOfElements;
};

#endif
