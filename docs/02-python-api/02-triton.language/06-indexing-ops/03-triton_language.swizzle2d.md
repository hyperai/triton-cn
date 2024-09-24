---
title: triton_language.swizzle2d
---

```python
triton.language.swizzle2d(i, j, size_i, size_j, size_g)
```


将行主序的 *size_i* size_j 矩阵的索引转换为每组 size_g* 行的列主序矩阵的索引。


例如， 对 `size_i = size_j = 4` 和 `size_g = 2`，它将转换


```python
 [[0 , 1 , 2 , 3 ],
 [4 , 5 , 6 , 7 ],
 [8 , 9 , 10, 11],
 [12, 13, 14, 15]]
```


为


```python
[[0, 2,  4 , 6 ],
 [1, 3,  5 , 7 ],
 [8, 10, 12, 14],
 [9, 11, 13, 15]]
```


