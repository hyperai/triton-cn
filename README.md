# Triton 中文文档
[中文文档](https://triton.hyper.ai/)｜[了解更多](https://hyper.ai/)

Triton 是一种用于并行编程的语言和编译器，旨在提供一个基于 Python 的编程环境，以高效编写自定义 DNN 计算内核，并能够在现代 GPU 硬件上以最大吞吐量运行。

由于现有 Triton 相关的中文学习资料较为零散，不便于开发者系统性学习，我们在 GitHub 上创建了 Triton 文档翻译项目。

随着 Triton 官方文档的更新，中文文档也会进行同步修订，你可以：

- 学习 Triton 中文文档，为翻译不准确或有歧义的地方 [提交 issue](https://github.com/hyperai/triton-cn/issues) 或 [PR](https://github.com/hyperai/triton-cn/pulls)
- 参与开源协作、追踪文档更新，并认领文档翻译，成为 Triton 中文文档贡献者
- 加入 Triton 中文社区、结识志同道合的伙伴，并参与深入的讨论和交流。

衷心希望能够通过这个项目，为 Triton 中文社区的发展贡献一份绵薄之力。


## 参与贡献

本地开发服务器需先安装 Node.js 以及 [pnpm](https://pnpm.io/installation)。

```bash
pnpm install
pnpm start
```

迁移图片，将第三方的外部图片按其完整路径进行迁移，例如图片：

```md
![图片](https://raw.githubusercontent.com/tvmai/tvmai.github.io/main/images/relay/dataflow.png)
```

请将其保存在项目中的如下路径：

```
static/img/docs/tvmai/tvmai.github.io/main/images/relay/dataflow.png
```

然后在文档中替换为：

```md
![图片](/img/docs/tvmai/tvmai.github.io/main/images/relay/dataflow.png)
```

生成 HTML 文件 (Deprecated)

```bash
sphinx-build -b html docs build
```

## 创建新版本

如果当前版本为 `0.12.0`，想升到 `0.13.0`，那么你需要先保存当前版本

```bash
pnpm run docusaurus docs:version 0.12.0
```

然后编辑 `docusaurus.config.ts` 中 `versions.current.label` 为最新版本 `0.13.0`
