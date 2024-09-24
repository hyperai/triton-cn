---
title: triton_language.rand
---

```python
triton.language.rand(seed, offset, n_rounds: constexpr = 10)
```


给定 1 个 `seed` 标量和 1 个 `offset` 块，返回 1 个在 $$U(0,1)$$ 中的 float32 类型的随机块。 


**参数****：**

* **seed** - 用于生成随机数的种子。
* **offsets** - 用于生成随机数的偏移量。


