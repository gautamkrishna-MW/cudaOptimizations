
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void reductionKernel(float* inpArr, size_t inpSize, float *outputVal)
{
    size_t gTIdx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tIdx = threadIdx.x;

    // Shuffle instructions
    size_t warpSize = 32;
    float value = inpArr[gTIdx];

    // No chance of thread divergence as warps with 'full-data' satisfy if-condition.
    if (gTIdx < (warpSize * (inpSize / warpSize)))
    {
        for (int i = warpSize / 2; i > 0; i = i / 2)
        {
            value += __shfl_down_sync(-1, value, i);
        }
    }
    // Warps with partial number of threads dealing with data are sent into else-condition.
    else
    {
        int remNum = inpSize % warpSize;
        for (int i = 1; i < remNum; i = i * 2)
        {
            float tempValue = __shfl_down_sync(-1, value, i);
            if ((tIdx % warpSize) < (remNum - 1))
                value += tempValue;
        }
    }
    
    if (tIdx % warpSize == 0)
        atomicAdd(outputVal, value);
}

#define NUMEL 5000
int main()
{
    //float inputArray[NUMEL] = { 3.48,2.71,-7.64,-3.42,8.05,-2.59,-9.14,6.69,6.11,3.59,-9.32,6.18,2.17, 2.82,-2.7,-2.5,0.88,6.54,-4.06,4.15,-7.62,2.79,4.48,3.62,-0.96,-7.79,-2.49,3.,6.75,-9.65, -4.86,6.,-0.05,4.98,-1.33,9.18,-3.4,3.5,5.08,-7.57,1.28,9.12,-8.78,-4.59,-6.8,5.34,3.62,2.38,3.4,-2.04 };
    /*printf("%d", 110 % 32);
    return 0;*/

    //float inputArray[NUMEL] = { 4,5,2,6,8,9,9,1,8,8,0,3,8,3,8,8,8,6,4,6,4,9,5,6,6,8,4,0,8,7,5,0, 4, 0, 4, 4, 0, 6, 6, 1, 7, 5, 2, 2, 3, 5, 7, 0, 9, 2 };
    float inputArray[NUMEL];
    for (int i = 0; i < NUMEL; i++)
    {
        inputArray[i] = (rand()%100 - 50) / 25.f;
    }

    float sum = 0;
    for (int i = 0; i < NUMEL; i++)
    {
        sum += inputArray[i];
    }
    printf("CPU: %f\n", sum);

    float* devInpArr, *outputVal;
    size_t inpSize = NUMEL;
    cudaMalloc((void**)&devInpArr, NUMEL * sizeof(float));
    cudaMalloc((void**)&outputVal, sizeof(float));
    cudaMemset(outputVal, 0, sizeof(float));
    cudaMemcpy(devInpArr, inputArray, NUMEL * sizeof(float), cudaMemcpyHostToDevice);

    size_t threads = 512;
    size_t blocks = ((NUMEL / threads) + 1);
    reductionKernel << <blocks, threads >> > (devInpArr, inpSize, outputVal);

    float outValGPU = 0;
    cudaMemcpy(&outValGPU, outputVal, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU: %f, ", outValGPU);

    cudaFree(devInpArr);
    cudaFree(outputVal);
    return 0;
}
