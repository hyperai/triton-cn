---
title: triton_language.split
---

```python
triton.language.split(a)→ tuple[tensor, tensor]
```


将张量沿着其最后 1 个维度分成 2 部分，该维度的大小必须为 2。


例如，给定 1 个形状为 (4,8,2) 的张量，生成 2 个形状为 (4,8) 的张量。给定 1 个形状为 (2) 的张量，返回 2 个标量。


如果希望拆分成多个部分，可以多次调用这个函数（可能还需要调用 reshape 函数）。这反映了 Triton 中的约束，即张量必须具有 2 的幂次方大小。


Split 是 join 的逆操作。


**参数****：**

* **a** (*Tensor*) - 要被分割的张量。

这个函数也可作为 `tensor` 的成员函数调用，作为 `x.split(...)` 而不是 `split(x, ...)`。


