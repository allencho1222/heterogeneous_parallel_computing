#include <stdio.h>

#include <CL/opencl.h>

const char* read_kernel_from_file(const char* filename) {
  FILE *fp = fopen(filename, "rt");
  size_t length;
  char *data;
  if(!fp) return 0;

  // get file length
  fseek(fp, 0, SEEK_END);
  length = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  // read program source
  data = (char *)malloc(length + 1);
  fread(data, sizeof(char), length, fp);
  data[length] = '\0';
  return data;
}

void debugKernelSource(const cl_program& program, const cl_device_id& device_id) {
  size_t log_size;
  clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  char* build_log = (char* )malloc((log_size+1));
  clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
  build_log[log_size] = '\0';
  printf("--- Build log ---\n ");
  printf("%s\n", build_log);
  free(build_log);
}

void debug(cl_int success) {
  switch (success) {
    case CL_INVALID_PROGRAM_EXECUTABLE:
      printf("invalid program executable\n");
      break;
    case CL_INVALID_COMMAND_QUEUE:
      printf("invalid command queue\n");
      break;
    case CL_INVALID_KERNEL:
      printf("invalid kernel\n");
      break;
    case CL_INVALID_CONTEXT:
      printf("invalid context\n");
      break;
    case CL_INVALID_KERNEL_ARGS:
      printf("invalid kernel args\n");
      break;
    case CL_INVALID_WORK_DIMENSION:
      printf("invalid work dimension\n");
      break;
    case CL_INVALID_WORK_GROUP_SIZE:
      printf("invalid work group size\n");
      break;
    case CL_INVALID_WORK_ITEM_SIZE:
      printf("invalid work item size\n");
      break;
    case CL_INVALID_GLOBAL_OFFSET:
      printf("invalid global offset\n");
      break;
    case CL_OUT_OF_RESOURCES:
      printf("out of resources\n");
      break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      printf("object allocation failure\n");
      break;
    case CL_INVALID_EVENT_WAIT_LIST:
      printf("invalid event wait list\n");
      break;
    case CL_OUT_OF_HOST_MEMORY:
      printf("out of host memory\n");
      break;
    default:
      printf("others\n");
  }
}
