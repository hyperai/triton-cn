```python
triton.language.permute(input, *dims)
```


排列张量的维度。


**参数****：**

* **input** (*Block*) - 输入张量。

* **dims** - 所需的维度顺序。例如，(2, 1, 0) 将在一个 3D 张量中反转维度的顺序。

`dims` 可以作为一个元组或单独的参数被传递：

```python
# These are equivalent
# 这些是等价的
permute(x, (2, 1, 0))
permute(x, 2, 1, 0)
```


`trans()` 和这个函数等价，除了当 `dims` 为空的时候，它会尝试进行 (1,0) 的置换。


这个函数也可作为 `tensor` 的成员函数调用，例如 `x.permute(...)` 代替 `permute(x, ...)`。


