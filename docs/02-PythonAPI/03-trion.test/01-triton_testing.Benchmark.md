```python
class triton.testing.Benchmark(self, x_names: List[str], x_vals: List[Any], line_arg: str, line_vals: List[Any], line_names: List[str], plot_name: str, args: Dict[str, Any], xlabel: str = '', ylabel: str = '', x_log: bool = False, y_log: bool = False, styles=None)
```


该类由 `perf_report` 函数调用，以简洁的 API 生成折线图。


```python
__init__(self, x_names: List[str], x_vals: List[Any], line_arg: str, line_vals: List[Any], line_names: List[str], plot_name: str, args: Dict[str, Any], xlabel: str = '', ylabel: str = '', x_log: bool = False, y_log: bool = False, styles=None)
```


构造函数。`x_vals` 可以是标量列表或元组/列表列表。如果 `x_vals` 是标量列表，并且存在多个 `x_names`，则所有参数将具有相同的值。如果 `x_vals` 是由元组/列表组成的列表，则每个元素的长度应与 `x_names` 相同。


**参数****：**

* **x_names** (*List[str]*) - 应该出现在绘图 x 轴上的参数名称。

* **x_vals** (*List[Any]*) - 用于 `x_names` 中的参数的值列表。

* **line_arg** (*str*) - 参数名称，对于该参数，不同的值对应于图中的不同线条。

* **line_vals** (*List[Any]*) - 用于 `line_arg` 参数的值的列表。

* **line_names** (*List[str]*) - 不同线条的标签名称。

* **plot_name** (*str*) - 图表的名称。

* **args** (*Dict[str**,**Any]*) - 在整个基准测试期间保持固定的关键字参数字典。

* **xlabel** (*str*, *optional*) - 图表的 x 轴标签。

* **ylabel** (*str*, *optional*) - 图表的 y 轴标签。

* **x_log** (*bool*, *optional*) - x 轴是否应为对数刻度。

* **y_log** (*bool*, *optional*) - y 轴是否应为对数刻度。

* **styles** (*list[tuple[str, str]]*) - 一个元组列表，每个元组包含两个元素：颜色和线型。


**方法**


|**__init__(self, x_names, x_vals, line_arg, ...)**|构造函数|
|:----|:----|


