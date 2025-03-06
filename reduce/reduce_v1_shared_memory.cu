#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define THREAD_PER_BLOCK 256            // 每个block有256个线程
#define N 32 * 1024 * 1024              // 共N个元素进行规约求和


__global__ void reduce_v1(float* d_input, float* d_output)
{
    float *input_begin = d_input + blockDim.x * blockIdx.x;
    __shared__ float s_input[THREAD_PER_BLOCK];
    s_input[threadIdx.x] = input_begin[threadIdx.x];

    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (threadIdx.x % (2 * i) == 0)
            s_input[threadIdx.x] += s_input[threadIdx.x + i];
            __syncthreads();
    }

    if (threadIdx.x == 0)
        d_output[blockIdx.x] = s_input[0];
}

bool check(float *h_result, float *h_output, int n)
{
    for (int i = 0; i < n; i++)
    {
        if(abs(h_result[i] - h_output[i]) > 1e-4)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    // std::cout << "Hello Reduce!" << std::endl;
    float *h_input = (float *)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    int block_num = N / THREAD_PER_BLOCK;
    float *h_output = (float *)malloc((block_num) * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, (block_num) * sizeof(float));

    float *h_result = (float *)malloc((block_num) * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        // h_input[i] = (float)i;z
        h_input[i] = 2.0 * (float)drand48() - 1.0;
    }

    // cpu上计算
    for (int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; j++)
        {
            cur += h_input[i * THREAD_PER_BLOCK + j];
        }
        h_result[i] = cur;
    }

    // gpu上计算
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduce_v1<<<Grid, Block>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, (block_num) * sizeof(float), cudaMemcpyDeviceToHost);

    // check result
    if (check(h_result, h_output, block_num))
    {
        std::cout << "Success" << std::endl;
    } else 
    {
        for (int i = 0; i < block_num; ++i)
        {
            std::cout << h_output[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Fail" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}