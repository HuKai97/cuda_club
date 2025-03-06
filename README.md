1. 设计cuda算法是按照block设计的，写程序时按照thread写的
2. 索引很容易乱套，最好为每一个block设计一个更进一步的索引
3. 绘制一个清晰的block算法图是很有必要的

# 一、reduce
https://zhuanlan.zhihu.com/p/426978026


# 二、sgemm

# 三、elementwise

# 四、gemv

# 五、spmv



# 其他
```bash
ctrl + shift + p调出gcc 生成 运行

为什么cudaMalloc一定要传入2级指针？
原因：cudaMalloc的作用是在GPU设备的全局内存中分配指定大小的内存，原型：cudaMalloc(void **devPtr, size_t size)，这里需要把在设备上分配内存的起始地址存储到这个指针所指向的位置，d_input本身是一个指针，想要修改它的值就必须传入它的地址，所以需要传入二级指针
float *d_input;
cudaMalloc((void**)&d_input, N * sizeof(float));
```


调试方法
1. ctrl+shift+p调出gcc 生成 + 运行/调试
2. F5需要先配置launch.json
3. 安装插件: Nsight Visual Studio Code Edition + launch中修改program路径


一些概念
1. GPU算力
https://www.techpowerup.com/gpu-specs/
https://www.techpowerup.com/cpu-specs/
FLOPS 每秒浮点运算次数 = GPU核心数(CORES) * 单核主频(Boost Clock) * CPU单个周期浮点运算能力
以3060 Ti 8G为例  16.20TFLOPS = 4864  * 1665 MHz * 2 = 16197120 FLOPS

2. GPU带宽
一个计算平台每秒所能完成的内存交换量  单位Byte/s
bandwidth = 内存频率（Memory Clock） * prefetch * 内存位宽（Memory Bus） / 8
以3060 Ti 8G为例   448.0 GB/s = 1750 MHz * 16 * 256 bit / 8 = ？

带宽和吞吐量：带宽是理想情况下的数据传送速率，吞吐量是某一个时间通过某个网络的实际传送速率

3. thread线程、warp线程束、block线程块、grid网络
    1. thread：GPU计算中的最小执行单元，每个线程都有自己独立的程序计数器和寄存器状态，执行相同的程序代码，所有线程可以共同访问global memory
    2. warp：GPU硬件调度和执行的基本单位，一个线程束通常包含32个线程，当一个block被启动时，线程会被分为若干个线程束，线程束中的所有线程会以SIMT单指令多线程的方式执行相同的指令。这样可以提高硬件的利用率和计算效率
    3. block：一组线程的集合，通过共享shared memory进行通信和同步，可以是一维/二维/三维
    4. grid：block的集合，当启动一个cuda核函数时，会创建一个grid，grid中包含多个block，block中又包括多个线程，每32个线程组成一个warp，同时执行。
        * 一维网格和线程块定义
            ```cpp
            dim3 grid_size(10);       // 定义一个10个block的一维网格
            dim3 block_size(256);     // 每个block包含256个线程
            kernel_name<<<grid_size, block_size>>>(...) 
            ```
        * 二维网格和线程块定义,适合用于处理二维数据，如图像/矩阵
            ```cpp
            dim3 grid_size(10, 20);   // 定义一个10x20个block的二维网格
            dim3 grid_size(16, 16);   // 每个block是一个16x16的二维线程布局
            ```
        * 三维网格和线程块定义，适合用于处理三维的数据结构，如点云数据
            ```cpp
            dim3 grid_size(10, 20, 30);
            dim3 block_size(16, 16, 16);
            ```
4. warp divergence 线程束分化
英伟达GPU架构演进近十年，从费米到安培：https://zhuanlan.zhihu.com/p/413145211

一个warp的不同线程如果有可能进入到if else的不同分支，就会造成warp divergence

kernel函数中尽可能不要写if...else

优化: 在整个block范围内，让前面一般的线程工作，后面一般的线程不工作，这样就解决了warp divergence问题

5. bank conflict
知乎：【CUDA编程概念】一、什么是bank conflict？

针对**共享内存**：为了提高内存读写带宽，共享内存会被分割成32个等大小的内存块即bank。而一个warp有32个线程，所以相当于一个线程对应一个bank，而一个warp的32个线程通常是同时一起操作的，正好可以一次性同时从32个bank中操作内存，这也是shared memory速度快的原因。
bank冲突：一个warp中的**不同线程**操作**同一个bank**中的**不同数据**，造成整体效率变低，速度变慢；如果不同线程拿的是同一个bank的同一个数据，那么就会触发广播机制，不算bank冲突