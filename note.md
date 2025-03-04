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