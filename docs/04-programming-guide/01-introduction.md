---
title: 简介
---

## 背景简介

在过去的十年中，深度神经网络 (DNNs) 作为一种重要的机器学习模型，已经成为一项在许多领域都能实现最先进性能的关键技术，涵盖自然语言处理[「SUTSKEVER2014」](#参考文献)、计算机视觉 [「REDMON2016」](#参考文献)和计算神经科学[「LEE2017」](#参考文献)等。这些模型的强大之处在于它们的分层结构，由一系列参数化（例如卷积）和非参数化（例如修正线性）的*层*组成。尽管这种模式因计算成本昂贵而闻名，但也生成了大量高度可并行化的工作，特别适合 multi-core 和 many-core 处理器。

因此，图形处理单元 (GPUs) 已经成为探索和/或部署领域内新研究思路的廉价和可访问资源。CUDA 和 OpenCL 等几个通用 GPU 计算框架的发布进一步推动了这一趋势，使得开发高性能程序变得更加容易。然而，对于不能有效地使用预先存在的优化基元组合实现的计算任务，GPU 仍然极具挑战性，特别是在优化数据局部性和并行性方面。更糟糕的是，GPU 架构也在迅速演进和专业化，如 NVIDIA（以及最近的 AMD ）微架构中张量内核心增加。

DNNs 提供的计算机会 (computational opportunities) 与 GPU 编程的实际困难之间的矛盾，使得学术界和工业界对领域特定语言 (DSLs) 和编译器产生了极大兴趣。遗憾的是，这些系统无论是基于多面体机制（例如 Tiramisu [[BAGHDADI2021]](#参考文献)、Tensor Comprehensions [[VASILACHE2018]](#参考文献)）还是调度语言（例如 Halide [[JRK2013]](#参考文献)、TVM [[CHEN2018]](#参考文献)），其在相同算法下的灵活性和性能上仍然不及最佳手写计算内核，如 [[cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)](<[https://docs.nvidia.com/cuda/cublas/index.htm](https://docs.nvidia.com/cuda/cublas/index.html)l>)、[cuDNN](https://docs.nvidia.com/deeplearning/cudnn/api/index.html)或 [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)。

本项目的主要假设是：基于阻塞算法 [[LAM1991]](#参考文献) 的编程范式可以促进神经网络的高性能计算内核的构建。我们特别重新审视传统的「单程序多数据」(SPMD [[AUGUIN1983]](#参考文献)) 执行模型在 GPU 上的应用，并提出一种变体，其中程序而非线程 (threads) 将被分块。例如，在矩阵乘法中，CUDA 和 Triton 的区别如下：

CUDA 编程模型（标量程序，阻塞线程)

```plain
#pragma parallel
for(int m = 0; m < M; m++) {
    #pragma parallel
    for(int n = 0; n < N; n++) {
        float acc = 0;
        for(int k = 0; k < K; k++)
            acc += A[m, k] * B[k, n];
        C[m, n] = acc;
    }
}
```

![图片](/img/docs/ProgrammingGuide/Introduction-1.png)

Triton 编程模型 (Blocked Program, Scalar Threads)

```plain
// Triton Programming Model
#pragma parallel
for(int m = 0; m < M; m += MB) {
    #pragma parallel
    for(int n = 0; n < N; n += NB) {
        float acc[MB, NB] = 0;
        for(int k = 0; k < K; k += KB)
            acc += A[m:m+MB, k:k+KB] @ B[k:k+KB, n:n+NB];
        C[m:m+MB, n:n+NB] = acc;
    }
}


```

![图片](/img/docs/ProgrammingGuide/Introduction-2.png)

这种方法的关键优势在于它产生了块结构化的迭代空间，为程序员在实现稀疏操作时提供了比现有 DSL 更多的灵活性，同时允许编译器针对数据局部性和并行性进行积极优化。

## 挑战

我们提出的编程范式面临的主要挑战是工作调度，即如何有效地在现代 GPU 上执行每个程序实例的工作分配。为了解决这个问题，Triton 编译器大量使用*块级数据流分析*，这是一种基于目标程序的控制流和数据流结构静态调度迭代块的技术。结果显示，该系统实际上效果非常好：我们的编译器能够自动应用广泛的有趣优化（例如自动合并、线程调度、预取、自动矢量化、张量核心感知的指令选择、共享内存分配/同步、异步复制调度）。当然，要做到这一切并不简单；本指南的一个目的就是让您了解它的工作原理。

## 参考文献

- [[SUTSKEVER2014]](#背景简介) I. Sutskever 等人，“使用神经网络的序列到序列学习”，NIPS 2014
- [[REDMON2016]](#背景简介) J. Redmon 等人，“You Only Look Once: 统一的实时对象检测”，CVPR 2016
- [[LEE2017]](#背景简介) K. Lee 等人，“SNEMI3D 连接组挑战的超人精度”，ArXiV 2017
- [[BAGHDADI2021]](#背景简介) R. Baghdadi 等人，“Tiramisu: 用于表达快速和可移植代码的多面体编译器”，CGO 2021
- [[VASILACHE2018]](#背景简介) N. Vasilache 等人，“Tensor Comprehensions: 与框架无关的高性能机器学习抽象”，ArXiV 2018
- [[JRK2013]](#背景简介) J. Ragan-Kelley 等人，“Halide: 用于优化并行性、局部性和重计算的语言和编译器”，PLDI 2013
- [[CHEN2018]](#背景简介) T. Chen 等人，“TVM: 用于深度学习的自动化端到端优化编译器”，OSDI 2018
- [[LAM1991]](#背景简介) M. Lam 等人，“阻塞算法的缓存性能和优化”，ASPLOS 1991
- [[AUGUIN1983]](#背景简介) M. Auguin 等人，“Opsila: 用于数值分析和信号处理的先进 SIMD”，EUROMICRO 1983
