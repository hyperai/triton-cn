```python
triton.language.device_assert(cond, msg='')
```


在运行时从设备上断言该条件。要使这个断言生效，需要将环境变量 `TRITON_DEBUG` 设置为非 `0` 的值。


使用 Python 的 `assert` 语句等同于调用这个函数，不过第 2 个参数必须被提供且必须是 1 个字符串，例如  `assert pid == 0, "pid != 0"`。要使这个 `assert` 语句生效，必须设置环境变量。


```python
tl.device_assert(pid == 0)
assert pid == 0, f"pid != 0"
```


**参数****：**

* **cond** - 要断言的条件。必须是 1 个布尔张量。
* **msg** - 如果断言失败时要打印的消息。必须是 1 个字符串字面值。

