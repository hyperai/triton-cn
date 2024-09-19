```python
triton.language.sort(x, dim: constexpr | None = None, descending: constexpr = constexpr[0])
```


沿着指定维度对张量进行排序。 


**参数****：**

* **x** (*Tensor*) - 要排序的输入张量。
* **dim** (*int*, *可选*) - 用于对张量进行排序的维度。如果为 None，则沿张量的最后一个维度进行排序。目前仅支持按最后一个维度排序。
* **descending** (*bool*, *可选*) - 如果设置为 True，则张量按降序排序。如果设置为 False，则张量按升序排序。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.sort(...)` 而不是 `sort(x, ...)`。


