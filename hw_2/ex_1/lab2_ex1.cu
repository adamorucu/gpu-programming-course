#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  *(out + i) = *(in1 + i) + *(in2 + i);
}

//@@ Insert code to implement timer start

//@@ Insert code to implement timer stop


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);

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

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here
  const int TPB = 32;
  const int BPG = (inputLength + TPB - 1)/TPB;

  //@@ Launch the GPU Kernel here
  vecAdd<<<BPG, TPB>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(DataType), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  printf("\nReference:\n");
  for (int i = 0; i < inputLength; i++) {
    printf("%.3f ", resultRef[i]);
  }
  printf("\n");

  printf("\nGPU:\n");
  for (int i = 0; i < inputLength; i++) {
    printf("%.3f ", hostOutput[i]);
  }
  printf("\n");

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
