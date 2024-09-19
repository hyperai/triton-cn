```python
triton.language.reshape(input, *shape, can_reorder=False)
```


返回一个具有与输入相同元素数量，但具有所提供形状的张量。


**参数****：**

* **input** (*Block*) - 输入张量。
* **shape** - 新的形状。

`shape` 可以作为 1 个元组或作为单独的参数被传递：

```python
# These are equivalent
# 这些是等价的
reshape(x, (32, 32))
reshape(x, 32, 32)
```


这个函数也可作为 `tensor` 的成员函数调用，例如 `x.reshape(...)` 代替 `reshape(x, ...)`。


