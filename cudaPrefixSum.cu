void launch();
int main() {
    launch();
    return 0;
}

#define _USE_MATH_DEFINES
#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <string>
#undef NDEBUG
#include <assert.h>

#define TEST_CASES 10
#define DEFAULT_BLOCK_SIZE 1024
#define DIVIDE_BY_TWO 2

#define CUDA_CHECK(call)                                        \
do {                                                            \
        cudaError_t error = call;                               \
        if (error != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",       \
                    __FILE__, __LINE__,                         \
                    cudaGetErrorString(error));                 \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
} while(0)

__global__ void k_blockPrefixSum(int* inpArray, int inpArrSize, int warpSize, int* blockSum, int* outArray);
__global__ void k_globalPrefixSum(int* inpArray, int* offsetArray, int inpArrSize);

void launch() {

    // Setting input lengths
    int input_sizes[TEST_CASES] = { 5,10,20,38,53,64,73,85,91,100 };

    // Allocating max length and dims
    int max_length = 0;
    for (int i = 0; i < TEST_CASES; i++)
        max_length = max(max_length, input_sizes[i]);

    int* inpArrayPtr = (int*)malloc(max_length * sizeof(int));
    int* outArray_cpu = (int*)malloc(max_length * sizeof(int));
    int* outArray_gpu = (int*)malloc(max_length * sizeof(int));

    // Initializing stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocating device memory and copying host matrices to device
    int* inpArrayPtr_d = nullptr;
    int* outArrayPtr_d = nullptr;
    int* blockSumPtr_d = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&inpArrayPtr_d, max_length * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&outArrayPtr_d, max_length * sizeof(int), stream));

    // Allocating memory for block-sum intermediate output
    CUDA_CHECK(cudaMallocAsync((void**)&blockSumPtr_d, max_length * sizeof(int), stream));

    // Setting GPU Launch parameters using device properties 
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    int block_size = DEFAULT_BLOCK_SIZE;

    // Running test cases with random inputs
    srand(time(NULL));
    for (int t_iter = 0; t_iter < TEST_CASES; t_iter++) {

        // Set input length and dimensions
        int inpArraySize = input_sizes[t_iter];

        // Input data generation for image A in the interval (-1,1)
        for (int iter = 0; iter < inpArraySize; iter++)
            inpArrayPtr[iter] = 1; // (rand() % 100) - 50;

        // Host to device memcpy
        CUDA_CHECK(cudaMemcpyAsync(inpArrayPtr_d, inpArrayPtr, inpArraySize * sizeof(int), cudaMemcpyHostToDevice, stream));

        // Setting block-size according to the available shared-memory 
        int minSharedMemoryRequired = ceil(block_size / prop.warpSize) * sizeof(int);
        while (minSharedMemoryRequired > prop.sharedMemPerBlock) {
            block_size /= DIVIDE_BY_TWO;
            minSharedMemoryRequired = ceil(block_size / prop.warpSize) * sizeof(int);
        }

        // Block-size 
        dim3 blockDim(block_size);
        blockDim.x = min(blockDim.x, prop.maxThreadsDim[0]);

        // Grid-size 
        dim3 gridDim(ceil(inpArraySize / (float)block_size));
        gridDim.x = min(gridDim.x, prop.maxGridSize[0]);

        // Kernel launch to compute prefix-sum of every block
        void* args_blockSum[] = { &inpArrayPtr_d, &inpArraySize, &prop.warpSize, &blockSumPtr_d, &outArrayPtr_d };
        CUDA_CHECK(cudaLaunchKernel((void*)k_blockPrefixSum, gridDim, blockDim, args_blockSum, minSharedMemoryRequired, stream));

        // Kernel to compute prefix-sum block offsets
        dim3 gridDim_blockOffset(ceil(gridDim.x / (float)block_size));
        gridDim_blockOffset.x = min(gridDim_blockOffset.x, prop.maxGridSize[0]);
        int* nullPtrVar = nullptr;
        void* args_blockCumSum[] = { &blockSumPtr_d, &gridDim.x, &prop.warpSize, &nullPtrVar, &blockSumPtr_d };
        CUDA_CHECK(cudaLaunchKernel((void*)k_blockPrefixSum, gridDim_blockOffset, blockDim, args_blockCumSum, minSharedMemoryRequired, stream));

        // Kernel to compute the final prefix-sum of all the elements by adding the block offsets
        void* args_finalSum[] = { &outArrayPtr_d, &blockSumPtr_d, &inpArraySize, &blockSumPtr_d };
        CUDA_CHECK(cudaLaunchKernel((void*)k_globalPrefixSum, gridDim, blockDim, args_finalSum, 0, stream));

        // Copy the final output to CPU.
        CUDA_CHECK(cudaMemcpyAsync(outArray_gpu, outArrayPtr_d, inpArraySize * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Output Verification using CPU implementation
        outArray_cpu[0] = inpArrayPtr[0];
        for (int i = 1; i < inpArraySize; i++) {
            outArray_cpu[i] = inpArrayPtr[i] + outArray_cpu[i - 1];
        }

        // Verification
        for (int i = 0; i < inpArraySize; i++) {
            if (outArray_cpu[i] != outArray_gpu[i])
                printf("%d\n", i);
            assert(outArray_cpu[i] == outArray_gpu[i]);
        }        
    }

    free(inpArrayPtr);
    free(outArray_cpu);
    free(outArray_gpu);

    cudaFreeAsync(inpArrayPtr_d, stream);
    cudaFreeAsync(outArrayPtr_d, stream);
    cudaFreeAsync(blockSumPtr_d, stream);
    cudaStreamDestroy(stream);
}

__global__ void k_globalPrefixSum(int* inpArray, int* offsetArray, int inpArrSize) {
    
    // Kernel adding the block offsets to all the elements to get the final prefix-sum output.
    int lIdx = threadIdx.x;
    for (int tIdx = (blockIdx.x * blockDim.x) + lIdx; tIdx < inpArrSize; tIdx += gridDim.x * blockDim.x)
        inpArray[tIdx] += offsetArray[blockIdx.x - 1] * (blockIdx.x > 0);
}

__global__ void k_blockPrefixSum(int* inpArray, int inpArrSize, int warpSize, int* blockSum, int* outArray) {
    
    // Allocating dynamic shared memory
    extern __shared__ int shMem[];
    int lIdx = threadIdx.x;
    size_t mask = __activemask();

    for (int tIdx = (blockIdx.x * blockDim.x) + lIdx; tIdx < inpArrSize; tIdx += gridDim.x * blockDim.x) {

        // Warp-level: Compute prefix-sum of every warp, store the last element of every warp - offset.
        // Compute the prefix-sum of stored offsets and add previous warp's offset to every element in the present 
        // warp. This gives block-level sum.
        
        // Block-level: Compute prefix-sum of all the last threads of all blocks, and add the previous block offset
        // to all the elements in present block. This results in global prefix-sum.

        // Warp Level prefix-sum computation
        outArray[tIdx] = inpArray[tIdx];
        int sizeLim = (int)ceil(log2((double)min(inpArrSize, warpSize)));
        int warpIdx = lIdx / warpSize;
        int wTIdx = (lIdx % warpSize);

        // thVal_curr is current thread's value
        // thVal_prev is the thVal_curr value of lower-ID thread given y stride
        int thVal_curr = outArray[tIdx];
        int thVal_prev = 0;

        for (int i = 0; i < sizeLim; i++) {
            int stride = 1 << i;
            thVal_prev = __shfl_up_sync(mask, thVal_curr, stride);

            // Instead of a conditional, the condition is multiplied to avoid divergence
            // tmpVal = (wTIdx >= stride)? thVal_prev : 0
            // thVal_curr += tmpVal
            thVal_curr += (thVal_prev * (wTIdx >= stride));
        }
        outArray[tIdx] = thVal_curr;

        // Every warp computes prefix-sum and updates the shared memory with the warp's last thread value
        if ((tIdx + 1) % warpSize == 0)
            shMem[warpIdx] = outArray[tIdx];

        __syncthreads();

        // Block Level prefix-sum computation
        int shMemSize = blockDim.x / warpSize;
        sizeLim = (int)ceil(log2((double)shMemSize));
        mask &= ((1U << shMemSize) - 1);
        if (lIdx < shMemSize) {
            thVal_curr = shMem[warpIdx + wTIdx];

            for (int i = 0; i < sizeLim; i++) {
                int stride = 1 << i;
                thVal_prev = __shfl_up_sync(mask, thVal_curr, stride);

                // Instead of a conditional, the condition is multiplied to avoid divergence
                // tmpVal = (wTIdx >= stride)? thVal_prev : 0
                // thVal_curr += tmpVal
                thVal_curr += (thVal_prev * (wTIdx >= stride));
            }
            shMem[warpIdx + wTIdx] = thVal_curr;
        }

        __syncthreads();

        // Add computed offsets in shared memory to all elements in the block (except first 32 threads)
        if (warpIdx > 0)
            outArray[tIdx] += shMem[warpIdx - 1];

        // Store the last thread's (of the block) value. This is an offset to next block.
        if (blockSum && (lIdx == blockDim.x - 1))
            blockSum[blockIdx.x] = outArray[tIdx];
    }
}
