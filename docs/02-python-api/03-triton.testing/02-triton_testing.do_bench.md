---
title: triton_testing.do_bench
---

```python
triton.testing.do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode='mean', device_type='cuda')
```


对所提供的函数的运行时间进行基准测试。默认情况下，返回函数 `fn` 的中位数运行时间，以及第 20 和第 80 百分位的性能。


**参数****：**

* **fn** (*Callable*) - 用于基准测试的函数。

* **warmup** (*int*) - 预热时间（以毫秒为单位）。

* **rep** (*int*) - 重复时间（以毫秒为单位）。

* **grad_to_none** (*torch.tensor*, *optional*) - 将提供张量的梯度重置为 None。

* **quantiles** (*list[float]*, *optional*) - 除中位数外要返回的性能百分位数。

* **fast_flush** *(bool,**default is**True)* - 在测量之间使用更快的内核来清除 L2 缓存。

* **return_mode** (*str*) - 要返回的统计量度。选项为「min」、「max」、「mean」或「median」。默认为「mean」。

