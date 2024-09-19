# triton.language.advance

```python
triton.language.advance(base, offsets)
```
推进 1 个块指针。

**参数****：**

* **base** - 要推进的块指针。
* **offsets** - 要推进的偏移量，是一个按维度划分的元组。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.advance(...)` 的方式而不是 `advance(x, ...)`。


