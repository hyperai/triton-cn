```python
triton.language.flip(x, dim=None)
```
沿着维度 *dim* 翻转张量 *x*。

**参数****：**

* **x** (*Block*) - 第 1 个输入张量。
* **dim** (*int*) - 要沿其翻转的维度（目前仅支持最后一个维度）。

这个函数也可作为 `tensor` 的成员函数调用，例如 `x.flip(...)` 而不是 `flip(x, ...)`。


