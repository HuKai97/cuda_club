#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define THREAD_PER_BLOCK 256            // 每个block有256个线程
#define N 32 * 1024 * 1024              // 共N个元素进行规约求和


__global__ void reduce(float *d_input, float *d_output)
{
    // 1、设计block
    // 3 1 7 0 4 1 6 3
    // 4   7   5   9
    // 11      14
    // 25
    // 2、为每个block设计一个起始索引  每个Block的第一个元素都是input_begin[0]
    float *input_begin = d_input + blockIdx.x * blockDim.x;
    // 3、写代码要从从线程角度思考
    // if (threadIdx.x == 0 or 2 or 4 or 6)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 1];
    // if (threadIdx.x == 0 or 4)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 2];
    // if (threadIdx.x == 0)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 4];
    // 4、如果线程不是8个，而是很多个呢？所以不能写成if  应该写成for
    // for (int i = 1; i < blockDim.x; i *= 2)
    // {
    //     if (threadIdx.x % (i * 2) == 0)
    //     {
    //         input_begin[threadIdx.x] += input_begin[threadIdx.x + i];
    //     }    
    // }
    // 5、同步问题：3+1计算完了，但是如果7+0还没计算完，就计算4+7 那么肯定会出问题
    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (threadIdx.x % (i * 2) == 0)
        {
            input_begin[threadIdx.x] += input_begin[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        // d_output跟block数量相同
        d_output[blockIdx.x] = input_begin[0];
    }
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
    reduce<<<Grid, Block>>>(d_input, d_output);
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