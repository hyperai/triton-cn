```python
triton.language.max_constancy(input, values)
```


告知编译器 `input` 中的首批值是常量。 


例如，如果 `values` 是 [4]，那么输入中每组 4 个值都应该是相等的，例如 [0, 0, 0, 0, 1, 1, 1, 1]。


