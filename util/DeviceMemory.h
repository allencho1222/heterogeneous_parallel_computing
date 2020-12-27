#ifndef DEVICE_MEMORY_H
#define DEVICE_MEMORY_H

#include <stdio.h>
#include <CL/opencl.h>

template <typename T>
class DeviceMemory {
public:
  // Constructor create memory buffer (e.g., cudaMalloc)
  DeviceMemory(cl_context* context, cl_command_queue* queue, 
               uint32_t numOfElements) {
    deviceMemory = clCreateBuffer(*context, 
                                  CL_MEM_READ_WRITE, 
                                  numOfElements * sizeof(T), // allocation size
                                  NULL, NULL);
    printf("Allocate %.2lf MB into device memory\n", 
        (double) (sizeof(T) * numOfElements) / (1024 * 1024));

    this->queue = queue;
    this->numOfElements = numOfElements;
  }
  ~DeviceMemory() {
    printf("Free device memory\n");
    free(deviceMemory);
  }

  // = operator write data into allocated memory (e.g., cudaMemcpy)
  void operator=(T* hostMemory) {
    // assuming synchronized malloc (CL_TRUE)
    clEnqueueWriteBuffer(*queue, deviceMemory, CL_TRUE, 0, 
                         sizeof(T) * numOfElements,
                         hostMemory,
                         0, NULL, NULL);
    printf("Copy %.2lf MB into device memory\n", 
        (double) (sizeof(T) * numOfElements) / (1024 * 1024));
  }

  cl_mem* getDeviceMemory() { return &(this->deviceMemory); }
  uint64_t getAllocSize() { return sizeof(T) * numOfElements; }

private:
  cl_mem deviceMemory;
  cl_command_queue* queue;

  uint32_t numOfElements;
};

#endif

