---
title: triton_language.static_print
---

```python
triton.language.static_print(*values, sep: str = ' ', end: str = '\n', file=None, flush=False)
```


在编译时打印数值。 参数与内置的 `print` 相同。


注意：调用 Python 内置的 `print` 不同于调用这个函数，它实际上映射到 `device_print`，对参数有特殊要求。


```python
tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
```


