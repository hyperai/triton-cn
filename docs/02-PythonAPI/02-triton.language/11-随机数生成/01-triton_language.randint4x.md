```python
triton.language.randint4x(seed, offset, n_rounds: constexpr = 10)
```


给定 1 个 `seed` 标量和 1 个 `offset` 块，返回 4 个 `int32` 类型的随机块。 


这是 Triton 的 Philox  伪随机数生成器的最高效入口点。


**参数****：**

* **seed** - 用于生成随机数的种子。
* **offsets** - 用于生成随机数的偏移量。


