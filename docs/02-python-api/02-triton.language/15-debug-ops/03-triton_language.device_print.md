---
title: triton_language.device_print
---

```python
triton.language.device_print(prefix, *args, hex=False)
```


在运行时从设备打印值。字符串格式化不适用于运行时的值，因此你应该将要打印的值作为参数提供。第一个值必须是字符串，所有后续的值必须是标量或张量。


调用 Python 内置的 `print` 等同于调用这个函数，并且参数的要求将与这个函数匹配（而不是普通的 `print` 要求）。


```python
tl.device_print("pid", pid)
print("pid", pid)
```


在 CUDA 中，printfs 的输出通过一个大小有限的 buffer 进行流传输在一个主机上，我们测量默认为 6912 KiB，但这可能在不同的 GPU 和 CUDA 版本上有所不同）。如果你注意到一些 printf 被丢弃了，你可以通过调用增加 buffer 大小。


```python
triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)
```


在运行使用 printf 的内核后，CUDA 可能会在尝试更改这个值时引发错误。在这里设置的值可能只影响当前设备（因此如果你有多个 GPU，你需要多次调用）。


**参数****：**

* **prefix** - 在值之前打印的前缀，必须是字符串字面值。
* **args** - 要打印的值，可以是任何张量或标量。
* **hex** - 以十六进制打印所有值，而不是十进制。

