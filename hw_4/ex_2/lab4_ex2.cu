
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define DataType double
#define NUM_STREAMS 4

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  *(out + i) = *(in1 + i) + *(in2 + i);
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

  int S_seg = 128;

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType*)malloc(inputLength*sizeof(DataType));
  hostInput2 = (DataType*)malloc(inputLength*sizeof(DataType));
  hostOutput = (DataType*)malloc(inputLength*sizeof(DataType));
  resultRef = (DataType*)malloc(inputLength*sizeof(DataType));
  
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

  // Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  //@@ Insert code to below to Copy memory to the GPU here
  for (int i = 0; i < inputLength; i += S_seg) {
    int streamIdx = i / S_seg % NUM_STREAMS;
    int segSize = min(S_seg, inputLength - i);

    cudaMemcpyAsync(&deviceInput1[i], &hostInput1[i], segSize*sizeof(DataType), cudaMemcpyHostToDevice, streams[streamIdx]);
    cudaMemcpyAsync(&deviceInput2[i], &hostInput2[i], segSize*sizeof(DataType), cudaMemcpyHostToDevice, streams[streamIdx]);

    vecAdd<<<(segSize + 31)/32, 32, 0, streams[streamIdx]>>>(&deviceInput1[i], &deviceInput2[i], &deviceOutput[i], segSize);

    cudaMemcpyAsync(&hostOutput[i], &deviceOutput[i], segSize*sizeof(DataType), cudaMemcpyDeviceToHost, streams[streamIdx]);
  }
  // Synchronize and clean up
  for (int i = 0; i < NUM_STREAMS; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time taken: %f ms\n", milliseconds);


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
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
