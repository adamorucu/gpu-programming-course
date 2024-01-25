
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define DataType double
#define NUM_STREAMS 4
#define TPB 64
#define S_SEG 128

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

  // int S_seg = 128;

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

  // dim3 dimGrid(ceil(S_seg / TPB));
  // dim3 dimBlock(TPB);

  int nSegments = (inputLength + S_SEG - 1) / S_SEG;
  int segmentSize = (inputLength + nSegments - 1) / nSegments;
  printf("segmentSize: %d\n", segmentSize);
  printf("nSegments: %d\n", nSegments);

  //@@ Insert code to below to Copy memory to the GPU here
  // for (int i = 0; i < inputLength; i += S_seg) {
  //   int streamIdx = i / S_seg % NUM_STREAMS;
  //   int segSize = min(S_seg, inputLength - i);

  //   cudaMemcpyAsync(&deviceInput1[i], &hostInput1[i], segSize*sizeof(DataType), cudaMemcpyHostToDevice, streams[streamIdx]);
  //   cudaMemcpyAsync(&deviceInput2[i], &hostInput2[i], segSize*sizeof(DataType), cudaMemcpyHostToDevice, streams[streamIdx]);

  //   // vecAdd<<<(segSize + 31)/32, 32, 0, streams[streamIdx]>>>(&deviceInput1[i], &deviceInput2[i], &deviceOutput[i], segSize);
  //   vecAdd<<<dimGrid, dimBlock, 0, streams[streamIdx]>>>(&deviceInput1[i], &deviceInput2[i], &deviceOutput[i], segSize);

  //   cudaMemcpyAsync(&hostOutput[i], &deviceOutput[i], segSize*sizeof(DataType), cudaMemcpyDeviceToHost, streams[streamIdx]);
  // }
    
  for (int i = 0; i < nSegments; i++)
  {
    int offset = i * segmentSize;
    int len = min(segmentSize, inputLength - offset);
    int blockSize = TPB;
    int gridSize = (len + blockSize - 1) / blockSize;
    cudaStream_t stream = streams[i % NUM_STREAMS];
    
    cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, len * sizeof(DataType), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, len * sizeof(DataType), cudaMemcpyHostToDevice, stream);


    vecAdd<<<gridSize, blockSize, 0, stream>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, len);

    cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, len * sizeof(DataType), cudaMemcpyDeviceToHost, stream);
  }
  
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
