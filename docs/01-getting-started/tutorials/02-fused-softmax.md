---
title: 融合 Softmax (Fused Softmax)
---

[在线运行此教程](https://openbayes.com/console/hyperai-tutorials/containers/QEhTxGYyzqY)

在本教程中，您将编写一个融合的 softmax 操作，该操作在某些类别的矩阵上比 PyTorch 的原生操作快得多：即那些可以适应 GPU 静态随机存取存储器 (SRAM) 的行。


通过这个过程，您将了解以下内容：


* 内核融合对于带宽受限操作的优势。
* Triton 中缩减操作。

## 动机


用于逐元素加法的自定义 GPU 内核有教学上的价值，但在实践中不能带来很大的进展。


让我们转而考虑一个简单的（数值稳定的）softmax 操作：


```python
import torch


import triton
import triton.language as tl
from triton.runtime import driver




def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch
    使用原生 PyTorch 计算 X 的逐行 softmax


    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    我们减去最大元素以避免溢出。Softmax 对于这种偏移是不变的。
    """
    # read  MN elements ; write M  elements
    # 读取 MN 个元素；写入 M 个元素
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    # 读取 MN + M 个元素；写入 MN 个元素
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    # 读取 MN 个元素；写入 MN 个元素
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    # 读取 MN 个元素；写入 M 个元素
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    # 读取 MN + M 个元素；写入 MN 个元素
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    # 总计：读取 5MN + 2M 个元素；写入 3MN + 2M 个元素
    return ret
```


直接在 PyTorch 中实现时，对于 $$ x \in \mathbb{R}^{M \times N} $$，计算 `y = naive_softmax(x)` 需要从 DRAM 中读取 $$ 5MN + 2M $$ 个元素，并写回 $$ 3MN + 2M $$ 个元素。



这显然是浪费的；我们更希望有一个自定义的「融合」内核，它只需读取一次 X，并在芯片上进行所有必要的计算。


这样做只需要读写 $$ MN $$ 字节，因此我们可以期望理论上的加速约为 4 倍（即 $$ \frac{8MN + 4M}{2MN} $$）。


`torch.jit.script` 标志旨在自动执行这种「内核融合」，但正如我们后面将看到的，它仍不够理想。


## 计算内核


softmax 内核工作原理如下：每个程序加载输入矩阵 X 的一组行，按程序数量跨步处理，对其进行归一化，并将结果写回输出 Y。


注意，Triton 的一个重要限制是每个块必须具有 2 的幂次数的元素，因此，如果我们要处理任意可能的输入形状，我们需要在内部「填充」每一行，并适当保护内存操作。


```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    # 程序起始行
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        # 步长表示我们需要对指针增加多少以推进 1 行
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # 块大小是大于 n_cols 的下一个二的幂，因此我们可以适配
        # row in a single block
        # 单个块中的行
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        # 将行加载到 SRAM 中，使用掩码，因为 BLOCK_SIZE 可能大于 n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        # 为了数值稳定性而减去最大值
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        # 请注意，Triton 中的指数运算速度很快，但是是近似的（例如，类似于 CUDA 中的 __expf）。
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        # 将输出写回 DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```


我们可以创建一个辅助函数，为任何给定的输入张量建立内核及其（元）参数队列。


```python
device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape


    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    # 每次循环迭代的块大小是大于 `x` 列数的最小二的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)


    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # 另一个技巧是通过增加每行分配的线程数来要求编译器使用更多的线程块 (`num_warps`)
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    # 将在下一个教程中看到如何以更自然的方式自动调整此值，以免自己进行手动启发式处理。
    num_warps = 8


    # Number of software piepling stages.
    # 软件流水线阶段的数量
    num_stages = 4 if SIZE_SMEM > 200000 else 2


    # Allocate output
    # 分配输出空间
    y = torch.empty_like(x)


    # pre-compile kernel to get register usage and compute thread occupancy.
    # 预编译内核以获取寄存器使用情况并计算线程占用情况。
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)


    num_programs = min(num_programs, n_rows)


    # Create a number of persistent programs.
    # 创建一些持久化程序。
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y
```


## 单元测试


我们将在一个具有不规则行和列数的矩阵上测试我们的内核。


这将验证我们的 Padding 机制是否起作用。


```python
torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
```


结果与预期相同。


## 基准测试


此处将基于输入矩阵中列数的函数进行基准测试，假设有 4096 行，定义  `naive_softmax` 。


然后将其性能与（1）`torch.softmax` 和（2）上面定义的 `naive_softmax` 进行比较。


```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot 用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name` `x_name` 的不同可能值
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot 参数名，其值对应于图表中不同线条
        line_vals=['triton', 'torch'],  # possible values for `line_arg`` `line_arg` 的可能值
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines 线条的标签名称
        styles=[('blue', '-'), ('green', '-')],  # line styles 线条的样式
        ylabel="GB/s",  # label name for the y-axis y 轴的标签名称
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot. 图表的名称，也用作保存图表的文件名
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name` `x_names` 和 `y_name` 中未包含的函数参数的值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)




benchmark.run(show_plots=True, print_data=True)
```

![图片](/img/docs/Tutorials/FusedSoftmax/02.png)

Out:

> softmax-performance:

|**N**|**Triton**|**Torch**|
|:----|:----|:----|
|  256.0  |  475.581977|  708.619322|
|  384.0  |  619.872425|  812.799315|
|  512.0  |  752.326527|  927.222924|
|  640.0  |  788.217790|  946.386719|
|  768.0  |  880.887679| 1014.912158|
|  896.0  |  937.344158| 1074.519017|
| 1024.0  |  994.049328| 1120.599053|
| 1152.0  | 1096.160464|  616.484209|
| 1280.0  | 1136.037680|  669.424776|
| 1408.0  | 1150.661622|  725.262518|
| 1536.0  | 1195.385896|  783.556680|
| 1664.0  | 1218.037815|  812.802866|
| 1792.0  | 1240.453775|  857.206087|
| 1920.0  | 1249.594057|  910.379759|
| 2048.0  | 1281.002942|  960.369226|
| 2176.0  | 1258.141618|  976.327061|
| 2304.0  | 1268.029374| 1013.671493|
| 2432.0  | 1295.384387| 1059.587886|
| 2560.0  | 1306.614187| 1084.683454|
| 2688.0  | 1317.169033| 1104.769558|
| 2816.0  | 1327.217578| 1127.015242|
| 2944.0  | 1321.850846| 1164.100210|
| 3072.0  | 1351.140419| 1185.534776|
| 3200.0  | 1355.270950| 1195.189132|
| 3328.0  | 1350.926797| 1219.700403|
| 3456.0  | 1370.851095| 1249.846232|
| 3584.0  | 1370.733345| 1257.045186|
| 3712.0  | 1380.222691| 1272.332674|
| 3840.0  | 1386.847005| 1304.931759|
| 3968.0  | 1390.096765| 1314.800917|
| 4096.0  | 1395.691852| 1329.296474|
| 4224.0  | 1336.892109| 1157.837774|
| 4352.0  | 1338.490269| 1173.375508|
| 4480.0  | 1350.203136| 1183.423201|
| 4608.0  | 1361.692557| 1198.281856|
| 4736.0  | 1359.538511| 1196.113447|
| 4864.0  | 1374.159725| 1224.748171|
| 4992.0  | 1370.339012| 1237.542346|
| 5120.0  | 1371.061881| 1250.239195|
| 5248.0  | 1373.741013| 1256.002531|
| 5376.0  | 1382.862170| 1286.354639|
| 5504.0  | 1377.679797| 1300.142739|
| 5632.0  | 1378.558008| 1311.940458|
| 5760.0  | 1393.962179| 1329.921162|
| 5888.0  | 1395.824888| 1346.085280|
| 6016.0  | 1401.488037| 1355.059080|
| 6144.0  | 1406.345907| 1374.157489|
| 6272.0  | 1412.687019| 1376.883517|
| 6400.0  | 1415.309106| 1389.410912|
| 6528.0  | 1417.204727| 1392.583463|
| 6656.0  | 1422.082775| 1405.407043|
| 6784.0  | 1416.999653| 1415.459830|
| 6912.0  | 1427.997548| 1424.580919|
| 7040.0  | 1420.079821| 1433.713238|
| 7168.0  | 1428.226868| 1434.182051|
| 7296.0  | 1426.907241| 1443.570904|
| [.0  | 1431.245969| 1444.524696|
| 7552.0  | 1429.852775| 1455.236120|
| 7680.0  | 1438.222846| 1459.114601|
| 7808.0  | 1432.084205| 1467.194446|
| 7936.0  | 1435.612336| 1467.986631|
| 8064.0  | 1434.118461| 1472.734245|
| 8192.0  | 1442.312192| 1483.740088|
| 8320.0  | 1388.784296| 1401.371945|
| 8448.0  | 1380.648971| 1407.791889|
| 8576.0  | 1397.384833| 1396.228603|
| 8704.0  | 1393.329000| 1400.798649|
| 8832.0  | 1382.315605| 1401.590681|
| 8960.0  | 1396.830311| 1413.000037|
| 9088.0  | 1409.916407| 1418.336829|
| 9216.0  | 1406.385628| 1423.454382|
| 9344.0  | 1399.874528| 1424.049331|
| 9472.0  | 1399.237655| 1435.753027|
| 9600.0  | 1397.459628| 1430.972090|
| 9728.0  | 1397.891660| 1440.957731|
| 9856.0  | 1413.115159| 1443.096680|
| 9984.0  | 1403.193995| 1448.557274|
|10112.0  | 1410.619129| 1460.444766|
|10240.0  | 1419.479137| 1469.013209|
|10368.0  | 1410.951646| 1462.658968|
|10496.0  | 1418.695729| 1464.967769|
|10624.0  | 1408.428065| 1471.324774|
|10752.0  | 1406.421698| 1472.142310|
|10880.0  | 1400.046054| 1480.109648|
|11008.0  | 1420.714401| 1480.192162|
|11136.0  | 1419.632677| 1486.624982|
|11264.0  | 1431.403761| 1485.485510|
|11392.0  | 1414.297153| 1487.808132|
|11520.0  | 1424.387130| 1492.888886|
|11648.0  | 1420.436500| 1499.220597|
|11776.0|1426.911001|1499.184|


在上面的图中，我们可以看到：


* Triton 比 Torch JIT 快 4 倍。这证实了我们对 Torch JIT 在这里没有进行任何融合的怀疑。
* 除了更易于阅读、理解和维护外，Triton 明显比 `torch.softmax` 快。

[Download Jupyter notebook: 02-fused-softmax.ipynb](https://triton-lang.org/main/_downloads/034d953b6214fedce6ea03803c712b89/02-fused-softmax.ipynb)

[Download Python source code: 02-fused-softmax.py](https://triton-lang.org/main/_downloads/d91442ac2982c4e0cc3ab0f43534afbc/02-fused-softmax.py)

[Download zipped: 02-fused-softmax.zip](https://triton-lang.org/main/_downloads/f66de4fbee2c4ba20b6f7f3ae99f7de3/02-fused-softmax.zip)
