#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
// #define NUM_BINS 64

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

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
    extern __shared__ unsigned int local_bins[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    int tx = threadIdx.x;

    // Initialize shared memory
    if (tx < num_bins) {
        local_bins[tx] = 0;
    }
    __syncthreads();

    // Update local histogram
    if (i < num_elements) {
        unsigned int bin = input[i];
        if (bin < num_bins) {
            atomicAdd(&local_bins[bin], 1);
        }
    }
    __syncthreads();

    // Combine local histograms into global histogram
    if (tx < num_bins) {
        atomicAdd(&bins[tx], local_bins[tx]);
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
    for (int i = 0; i < length; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;



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
  hostInput = (unsigned int*)malloc(inputLength*sizeof(unsigned int));
  hostBins = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
  resultRef = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));

  if (hostBins == NULL)
    fprintf(stderr, "Failed to allocate memory for hostBins\n");
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  srand(time(NULL));
  for (unsigned int i = 0; i < inputLength; i++) {
    hostInput[i] = rand() % NUM_BINS;
  }

  //@@ Insert code below to create reference result in CPU
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
  printf("CUDA Malloc\n");
  cudaMalloc(&deviceInput, inputLength*sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS*sizeof(unsigned int));
  // cudaMalloc(&deviceLocalBins, NUM_BINS*sizeof(unsigned int));

  if (deviceBins == NULL)
    fprintf(stderr, "Failed to allocate memory for deviceBins\n");

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyHostToDevice);
  // cudaMemcpy(deviceLocalBins, hostBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS*sizeof(unsigned int));
  // cudaMemset(deviceLocalBins, 0, NUM_BINS*sizeof(unsigned int));
  //@@ Initialize the grid and block dimensions here
  int TPB = 64;
  int BPG = (inputLength + TPB - 1)/TPB;

  //@@ Launch the GPU Kernel here
  // histogram_kernel<<<BPG, TPB>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  const size_t smemSize = TPB*sizeof(unsigned int);
  histogram_kernel<<<BPG, TPB, smemSize>>>(deviceInput, deviceBins, inputLength, NUM_BINS);


  //@@ Initialize the second grid and block dimensions here
  TPB = 32;
  BPG = (inputLength + TPB - 1)/TPB;

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<BPG, TPB>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference

  printf("\nReference:\n");
  for (int i = 0; i < NUM_BINS; i++) {
    printf("%d ", resultRef[i]);
  }
  printf("\n");
  
  printf("\nGPU:\n");
  for (int i = 0; i < NUM_BINS; i++) {
    printf("%d ", hostBins[i]);
  }
  printf("\n");

  bool identical = identicalArr(resultRef, hostBins, NUM_BINS);
  if (identical)
    printf("Identical arrays\n");
  else
    printf("NOT IDENTICAL\n");
  
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  // cudaFree(deviceLocalBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

