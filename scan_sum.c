// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

// implement the work-efficient scan kernel to generate per-block scan array and store the block sums into an auxiliary block sum array.
// use shared memory to reduce the number of global memory accesses, handle the boundary conditions when loading input list elements into the shared memory
// reuse the kernel to perform scan on the auxiliary block sum array to translate the elements into accumulative block sums. Note that this kernel will be launched with only one block.
// implement the kernel that adds the accumulative block sums to the appropriate elements of the per-block scan array to complete the scan for all the elements.

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                      \
    do                                                                     \
    {                                                                      \
        cudaError_t err = stmt;                                            \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
            return -1;                                                     \
        }                                                                  \
    } while (0)

__global__ void scan(float *input, float *output, int len, float *sum)
{
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from the host
    __shared__ float T[BLOCK_SIZE * 2];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int idx = 2 * bx * BLOCK_SIZE + tx;
    T[tx] = (idx < len) ? input[idx] : 0;
    T[tx + BLOCK_SIZE] = (idx + BLOCK_SIZE < len) ? input[idx + BLOCK_SIZE] : 0;

    int stride = 1;
    while (stride < 2 * BLOCK_SIZE)
    {
        __syncthreads();
        int index = (tx + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE && (index - stride) >= 0)
            T[index] += T[index - stride];
        stride = stride * 2;
    }

    //post scan
    stride = BLOCK_SIZE / 2;
    while (stride > 0)
    {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if ((index + stride) < 2 * BLOCK_SIZE)
            T[index + stride] += T[index];
        stride = stride / 2;
    }

    __syncthreads();

    if (idx < len)
        output[idx] = T[tx];
    if (idx + BLOCK_SIZE < len)
        output[idx + BLOCK_SIZE] = T[tx + BLOCK_SIZE];

    if (tx == BLOCK_SIZE - 1)
    {
        sum[bx] = T[2 * BLOCK_SIZE - 1];
    }
}

__global__ void add(float *sum, float *output, int len)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int i = 2 * bx * BLOCK_SIZE + tx;
    if (bx > 0 && i < len)
    {
        output[i] += sum[bx - 1];
    }
    if (bx != 0 && i + BLOCK_SIZE < len)
    {
        output[i + BLOCK_SIZE] += sum[bx - 1];
    }
}

int main(int argc, char **argv)
{
    wbArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    float *sum;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float *)malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ",
          numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void **)&sum, ceil(1.0 * numElements / (BLOCK_SIZE << 1)) * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                       cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(ceil((1.0 * numElements) / (BLOCK_SIZE << 1)), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce

    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, sum);
    dim3 G(1, 1, 1);
    scan<<<G, dimBlock>>>(sum, sum, ceil(1.0 * numElements / (BLOCK_SIZE << 1)), deviceInput);
    add<<<dimGrid, dimBlock>>>(sum, deviceOutput, numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                       cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(sum);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
