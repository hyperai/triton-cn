---
title: triton_testing.assert_close
---

```python
triton.testing.assert_close(x, y, atol=None, rtol=None, err_msg='')
```


断言两个输入在一定公差范围内接近。


**参数****：**

* **x** (*scala*, *list*, *numpy.ndarray*, *or* *torch.Tensor*) - 第一个输入。

* **y** (*scala*, *list*, *numpy.ndarray*, *or* *torch.Tensor*) - 第二个输入。

* **atol** (*float*, *optional*) - 绝对公差。默认值为 1e-2。

* **rtol** (*float*, *optional*) - 相对公差。默认值为 0。

* **err_msg** (*str*) - 如果断言失败要使用的错误消息。

