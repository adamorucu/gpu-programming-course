# Assignment 1

## Exercise 1 - Reflections on GPU-accelerated Computing

### 1
- CPUs are faster and better at handling complex instructions
- GPUs have a lot more cores which means they can work on a lot of tasks in parallel
- CPUs usually have higher clock speeds and there fore can execute a single instruction faster than a GPU

### 2 

| Rank | Name | GPU model | Rpeak / Power (TFlops/kW) |
|---|---|---|---|
| 1 | Frontier | AMD Instinct MI250X | 73.95 |
| 2 | Fugaku | Fujitsu A64FX | 14.78 |
| 3 | LUMI | AMD Instinct MI250X | 51.36 |
| 4 | Leonardo | Nvidia Ampere A100 | 32.14 |
| 5 | Summit | Nvidia Tesla V100 / Volta GV100 | 14.66 |
| 6 | Sierra | Nvidia Tesla V100 / Volta GV100 | 12.64 |
| 7 | Sunway TaihuLight | - | 6.05 |
| 8 | Perlmutter | Nvidia Ampere A100 | 27.04 |
| 9 | Selene | Nvidia Ampere A100 | 23.81 |
| 10 | Tianhe-2A | Matrix-2000 | 3.30 |

- 9 out of 10 have a GPU
- Out of those 5 are by Nvidia 2 by AMD and one each from Fujitsu and Matrix
- source: https://www.top500.org/lists/top500/2023/06/

## Exercise 2 - Query Nvidia GPU Compute Capability
![deviceQuery](deviceQuery.png)

The compute capability is 7.5.

- Memory Clock rate: 5001 Mhz
- Bus width: 256-bit
- DDR: 2
- Memory bandwith: 5001 * 256 / 8 / 1024 * 2 = 312 GB/s

![bandwidthTest](bandwidthTest.png)

The bandiwth from the test is lower than the value calculated.

## Exercise 3 - Rodinia CUDA benchmarks and Comparison with CPU

I've carried out experiments the heartwall and k-means tests. Changing the Makefile was not necessary however I removed the unnecessary file complitations to simplify the process and shorten the compilcation times. The two selected tasks are imaging and data mining tasks which can be done in parallel. Looking at the results in the below figures it can be seen that for both tests the execution time is more than twice as fast on a GPU compared to a CPU. While both executions can be parallelised the OMP code has been run using a single thread while the CUDA code uses 256 threads per block for the  imaging task. This is means that it can do many more calculations at the same time.

![heartwall](heartwall.png)

![kmeans](kmeans.png)

## Exercise 4 - Run a HelloWorld on AMD GPU

To launch the code on the AMD GPUs in Dardel. First, one needs to get a GPU allocation. Then the executable created can be launched on the GPU using the `srun` command and specifying which node the program should run on with the `-n` flag.

![dardel](dardel.png)