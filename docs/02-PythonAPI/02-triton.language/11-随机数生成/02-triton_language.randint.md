```python
triton.language.randint(seed, offset, n_rounds: constexpr = 10)
```


给定 1 个 `seed` 标量和 1 个 `offset` 块，返回 1 个 `int32` 类型的随机块。 


如果需要多个随机数流，使用 randint4x 可能比连续调用 4 次 randint 更快。


**参数****：**

* **seed** - 用于生成随机数的种子。
* **offsets** - 用于生成随机数的偏移量。


