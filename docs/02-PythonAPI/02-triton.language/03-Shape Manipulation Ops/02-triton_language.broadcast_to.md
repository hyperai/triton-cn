```python
triton.language.broadcast_to(input, *shape)
```


尝试将给定的张量广播到新的 `shape`。


**参数****：**

* input (Block) - 输入张量。
* shape - 所需的形状。

`shape` 可以以 1 个元组或独立参数被传入：

```python
# These are equivalent 
# 这些是等效的
broadcast_to(x, (32, 32))
broadcast_to(x, 32, 32)
```


该函数也可作为 `tensor` 的 1 个成员函数调用，使用 `x.broadcast_to(...)` 而不是 `broadcast_to(x, ...)`。


