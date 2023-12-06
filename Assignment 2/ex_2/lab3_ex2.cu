
#include <stdio.h>
#include <sys/time.h>

#define DataType double

int divUp(int a, int b) { return (a + b - 1) / b; }

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  if (numAColumns != numBRows)
    return;
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // column
  const int r = blockIdx.y * blockDim.y + threadIdx.y; // row
  
  if ((c >= numBColumns) || (r >= numARows))
    return;

  *(C + r*numBColumns + c) = 0;
  for (int k=0; k<numAColumns; k++) {
    *(C + r*numBColumns + c) += *(A + r*numAColumns + k) * *(B + k*numBColumns + c);
  }
}

__host__ void gemm_host(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  if (numAColumns != numBRows)
    return;

  for (int r=0; r<numARows; r++) {
    for (int c=0; c<numBColumns; c++) {
      *(C + r*numBColumns + c) = 0;
      for (int k=0; k<numAColumns; k++) {
        *(C + r*numBColumns + c) += *(A + r*numAColumns + k) * *(B + k*numBColumns + c);
      }
    }
  }
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType*)malloc(numAColumns*numARows*sizeof(DataType));
  hostB = (DataType*)malloc(numBColumns*numBRows*sizeof(DataType));
  hostC = (DataType*)malloc(numARows*numBColumns*sizeof(DataType));
  resultRef = (DataType*)malloc(numARows*numBColumns*sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  srand((unsigned)time(NULL));
  for (int r=0; r<numARows; r++) {
    for (int c=0; c<numAColumns; c++) {
      *(hostA + r*numAColumns + c) = (DataType)rand() / RAND_MAX * 100;
    }
  }
  for (int r=0; r<numBRows; r++) {
    for (int c=0; c<numBColumns; c++) {
      *(hostB + r*numBColumns + c) = (DataType)rand() / RAND_MAX * 100;
    }
  }
  for (int r=0; r<numCRows; r++) {
    for (int c=0; c<numCColumns; c++) {
      *(hostC + r*numCColumns + c) = 0;
    }
  }
  gemm_host(hostA, hostB, resultRef, numARows, numAColumns, numBRows, numBColumns);

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows*numAColumns*sizeof(DataType));
  cudaMalloc(&deviceB, numBRows*numBColumns*sizeof(DataType));
  cudaMalloc(&deviceC, numARows*numBColumns*sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, numARows*numBColumns*sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 blockSize(8, 8);
  dim3 gridSize(divUp(numARows, 8), divUp(numBColumns, 8));

  //@@ Launch the GPU Kernel here
  gemm<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostA, deviceA, numARows*numAColumns*sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostB, deviceB, numBRows*numBColumns*sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostC, deviceC, numARows*numBColumns*sizeof(DataType), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  printf("Reference Matrix:\n");
    for (int r = 0; r < numARows; r++) {
        for (int c = 0; c < numBColumns; c++) {
            printf("%.1f ", resultRef[r * numBColumns + c]);
        }
        printf("\n");
    }

  printf("GPU Matrix:\n");
    for (int r = 0; r < numARows; r++) {
        for (int c = 0; c < numBColumns; c++) {
            printf("%.1f ", hostC[r * numBColumns + c]);
        }
        printf("\n");
    }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
