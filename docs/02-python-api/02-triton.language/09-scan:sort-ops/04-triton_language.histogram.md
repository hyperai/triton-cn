---
title: triton_language.histogram
---

```python
triton.language.histogram(input, num_bins)
```


基于 `input` 张量计算 1 个具有 num_bins 个 bin 的直方图，每个 bin 宽度为 1，起始于 0。 


**参数：**

* **input** (*Tensor*) - 输入张量。
* **num_bins** (*int*) - 直方图的箱数。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.histogram(...)` 而不是 `histogram(x, ...)`。


