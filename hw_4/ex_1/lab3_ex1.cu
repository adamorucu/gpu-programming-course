#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define DataType unsigned int

__global__ void histogram_kernel2(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
  int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        unsigned int bin = input[i];
        if (bin < num_bins) {
            atomicAdd(&bins[bin], 1);
        }
    }
}

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins) {
  __shared__ DataType local_bins[NUM_BINS];
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < num_elements) {
    atomicAdd(&local_bins[input[idx]], 1);
  }
  __syncthreads();

  int shared = ((int)num_bins/blockDim.x);
  for (int i = shared*threadIdx.x ; i < shared*threadIdx.x + shared ; i++) {
    atomicAdd(&bins[i], local_bins[i]);
    local_bins[i] = 0;
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
//@@ Insert code below to clean up bins that saturate at 127
 int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < num_bins) {
        bins[i] = min(bins[i], 127u);
    }
}


bool identicalArr(unsigned int *arr1, unsigned int *arr2, int length) {
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
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;
  // unsigned int *deviceLocalBins;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  printf("cpu Malloc\n");
  hostInput = (DataType*)malloc(inputLength*sizeof(DataType));
  hostBins = (DataType*)malloc(NUM_BINS*sizeof(DataType));
  resultRef = (DataType*)malloc(NUM_BINS*sizeof(DataType));

  if (hostBins == NULL)
    fprintf(stderr, "Failed to allocate memory for hostBins\n");
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  srand(time(NULL));
  for (unsigned int i = 0; i < inputLength; i++) {
    hostInput[i] = rand() % NUM_BINS;
  }

  //@@ insert code below to create reference result in cpu
  memset(resultRef, 0, NUM_BINS * sizeof(unsigned int));
  for (unsigned int i = 0; i < inputLength; i++) {
    if (hostInput[i] < NUM_BINS) {
      resultRef[hostInput[i]]++;
    }
  }
  for (unsigned int i = 0; i < NUM_BINS; i++) {
    resultRef[i] = min(resultRef[i], 127u);
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(DataType));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(DataType));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS);

  //@@ Initialize the grid and block dimensions here
  int TPB = 1024;
  int BPG = (inputLength + TPB - 1)/TPB;

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<BPG, TPB>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<BPG, TPB>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(DataType), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  bool identical = identicalArr(resultRef, hostBins, NUM_BINS);
  if (identical)
    printf("Identical arrays\n");
  else
    printf("NOT IDENTICAL\n");

  printf("\nGPU:\n");
  for (int i = 0; i < NUM_BINS; i++) {
    printf("%d ", hostBins[i]);
  }
  printf("\n");

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);  
  return 0;
}