---
title: triton_language.join
---

```python
triton.language.join(a, b)
```

在 1 个新的次要维度中连接给定的张量。


For example, given two tensors of shape (4,8), produces a new tensor of shape (4,8,2). Given two scalars, returns a tensor of shape (2).


例如，给定 2 个形状为 (4,8) 的张量，生成 1 个新的形状为 (4,8,2) 的张量。给定 2 个标量，返回 1 个形状为 (2) 的张量。


2 个输入被广播到相同的形状。


If you want to join more than two elements, you can use multiple calls to this function. This reflects the constraint in Triton that tensors must have power-of-two sizes.


如果你想连接超过 2 个元素，可以多次调用这个函数。这反映了 Triton 中的约束，即张量的大小必须是 2 的幂。


`join` 是 `split` 的逆操作。


**参数****：**

* **a** (*T**ensor*)– 第 1 个输入张量。
* **b** (*T**en**sor*) - 第 2 个输入张量。

