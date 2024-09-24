---
title: triton_language.static_assert
---

```python
triton.language.static_assert(cond, msg='')
```


在编译时断言条件。 无需设置 `TRITON_DEBUG` 环境变量。


```python
tl.static_assert(BLOCK_SIZE == 1024)
```


