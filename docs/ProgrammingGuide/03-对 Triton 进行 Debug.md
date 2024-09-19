本教程提供了调试 Triton 程序的指导，主要面向 Triton 用户。有兴趣探索 Triton 后端，包括 MLIR 代码转换及 LLVM 代码生成的开发者，可以参考此 [部分](https://github.com/triton-lang/triton?tab=readme-ov-file#tips-for-hacking) 来查看调试选项。


## 使用 Triton debugging operations


Triton 包括 4 个 Debug 算子，允许用户检查和查看张量值：


* `static_print` 和 `static_assert` 用于编译时 Debug。
* `device_print` 和 `device_assert` 用于运行时 Debug。
* 当 `TRITON_DEBUG` 设置为 `1` 时，`device_assert` 执行。其他 Debug 算子会执行，且不受 `TRITON_DEBUG` 值影响。

## 使用解释器 (interpreter)


解释器 (interpreter) 是调试 Triton 程序的直接且有帮助的工具。它允许 Triton 用户在 CPU 上运行 Triton 程序，并检查每个操作的中间结果。要启用解释器模式，将环境变量 `TRITON_INTERPRET` 设置为 `1`。此设置使得所有 Triton 内核绕过编译，并由解释器使用 Triton 操作的 numpy 等效项进行模拟。解释器按顺序处理每个 Triton 程序实例，逐个执行操作。


有三种使用解释器的主要方式：


* 使用 Python `print` 函数打印每个操作的中间结果。要检查整个张量，请使用 `print(tensor)`。要查看索引处的单个张量值，请使用 `print(tensor.handle.data[idx])`。

* 附加 `pdb` 进行 Triton 程序的逐步调试：

```bash
TRITON_INTERPRET=1 pdb main.py
b main.py:<line number>
r
```


* 导入 pdb 包并在 Triton 程序中设置断点：

```python
import triton
import triton.language as tl
import pdb


@triton.jit
def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
  pdb.set_trace()
  offs = tl.arange(0, BLOCK_SIZE)
  x = tl.load(x_ptr + offs)
  tl.store(y_ptr + offs, x)
```

### 

### 局限性


解释器有一些已知局限性：


不支持对 `bfloat16` 数值类型的操作。要在 `bfloat16` 张量上执行操作，请使用 `tl.cast(tensor)` 将张量转换为 `float32`。


不支持间接内存访问模式，例如：


```python
ptr = tl.load(ptr)
x = tl.load(ptr)
```


## 使用第三方工具


对于在 NVIDIA GPU 上的调试，[compute-sanitizer](https://docs.nvidia.com/cuda/compute-sanitizer/index.html) 是一个有效的工具，用于检查数据竞争和内存访问问题。要使用它，请在运行 Triton 程序的命令前加上 `compute-sanitizer`。


对于在 AMD GPU 上的调试，可以尝试 ROCm 的 LLVM [AddressSanitizer](https://rocm.docs.amd.com/en/latest/conceptual/using-gpu-sanitizer.html)。


要详细可视化 Triton 程序的内存访问，考虑使用 [triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz) 工具，它与底层 GPU 无关。


