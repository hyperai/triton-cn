---
title: triton_language.arange
---

```python
triton.language.arange(start, end)
```


返回半开区间 `[start, end)` 内的连续值。`end - start` 必须小于等于 `TRITON_MAX_TENSOR_NUMEL = 1048576`。


**参数****：**

* **start** (*int32*) - 区间的起始值。必须是 2 的幂。

* **end** (*int32*) - 区间的结束值。必须是大于 `start` 的 2 的幂。

