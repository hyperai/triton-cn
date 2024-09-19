```python
triton.language.trans(input: tensor, *dims)
```


置换张量的维度。


如果参数`dims` 未被指定，该函数默认为 (1,0) 置换，有效地转置了 1 个二维张量。


**参数****：**

* **input** – 输入张量。
* **dims** – 期望的维度顺序。例如，(2, 1, 0) 将在 1 个三维张量中反转维度的顺序。

`dims` 可以作为 1 个元组或作为单独的参数传递：

```python
# These are equivalent
# 这些是等价的
trans(x, (2, 1, 0))
trans(x, 2, 1, 0)
```


`permute()` 和这个函数是等价的，但它不包含当没有指定置换时的特殊处理情况。


这个函数也可作为 `tensor` 的成员函数调用，使用 `x.trans(...)` 的方式，而不是 `trans(x, ...)`。


