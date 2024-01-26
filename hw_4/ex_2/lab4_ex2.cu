
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define DataType double
#define NUM_STREAMS 4
#define TPB 32
// #define S_SEG 128

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len, int offset) {
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x*blockDim.x + threadIdx.x + offset;
  if (i < len)
		out[i] = in1[i] + in2[i];
}

bool identicalArr(DataType *arr1, DataType *arr2, int length) {
    int counter = 0;
    bool identical = true;
    for (int i = 0; i < length; i++) {
        if (arr1[i] != arr2[i]) {
            printf("IND: %d\n", i);
            identical = false;
            counter++;
        }
    }
    return identical;
}

struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}
double cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
  return time;
}

int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;
  cudaStream_t streams[NUM_STREAMS];

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  int segSize = atoi(argv[2]);

  // int S_seg = 128;

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  cudaHostAlloc(&hostInput1, inputLength * sizeof(DataType), cudaHostAllocDefault);
  cudaHostAlloc(&hostInput2, inputLength * sizeof(DataType), cudaHostAllocDefault);
  cudaHostAlloc(&hostOutput, inputLength * sizeof(DataType), cudaHostAllocDefault);
  cudaHostAlloc(&resultRef, inputLength * sizeof(DataType), cudaHostAllocDefault);
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand((unsigned)time(NULL));
  for (int i=0; i < inputLength; i++) {
    hostInput1[i] = (DataType)rand() / RAND_MAX * 100;
    hostInput2[i] = (DataType)rand() / RAND_MAX * 100;
    *(resultRef + i) = *(hostInput1 + i) + *(hostInput2 + i);
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength*sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength*sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength*sizeof(DataType));

  // Start streams
  for (int i = 0; i < NUM_STREAMS; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  cputimer_start();
  for (int i = 0; i < inputLength; i+=segSize)
  {
    int len = min(segSize, inputLength - i);
    int gridSize = (len + TPB - 1) / TPB;
    cudaStream_t stream = streams[(i/segSize) % NUM_STREAMS];
    
    cudaMemcpyAsync(&deviceInput1[i], &hostInput1[i], len * sizeof(DataType), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(&deviceInput2[i], &hostInput2[i], len * sizeof(DataType), cudaMemcpyHostToDevice, stream);

    vecAdd<<<gridSize, TPB, 0, stream>>>(deviceInput1, deviceInput2, deviceOutput, inputLength, i);

    cudaMemcpyAsync(&hostOutput[i], &deviceOutput[i], len * sizeof(DataType), cudaMemcpyDeviceToHost, stream);
  }
  cputimer_stop("Calculation");
  
  for (int i = 0; i < NUM_STREAMS; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  //@@ Insert code below to compare the output with the reference
  bool identical = identicalArr(resultRef, hostOutput, inputLength);
  if (identical)
    printf("Identical arrays\n");
  else
    printf("NOT IDENTICAL\n");

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  cudaFreeHost(resultRef);

  return 0;
}
