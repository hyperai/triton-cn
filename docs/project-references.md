---
title: 引用项目
---

# DeepSpeed

https://github.com/microsoft/DeepSpeed

检查点是降低训练大型语言模型成本的关键技术，在训练过程中可以保存模型状态，避免从头开始训练。DeepSpeed 是一种通用检查点技术，是解决分布式检查点问题的最全面的解决方案。

- 灵活的检查点可沿任何训练并行技术（即 PP、TP、DP、ZeRO-DP、SP、MoE）重塑训练

- 弹性资源管理，在训练和微调中随意增加或减少硬件资源

- 支持多种商业规模模型的真实世界用例（例如 BLOOM、Megatron GPT、LLAMA、Microsoft Phi）

DeepSpeed 集成了 Triton，提高了 BERT 类模型在 float16 精度下的推理速度。为不同的模型和 GPU 实现了 1.14-1.68 倍的加速（或 12-41% 的延迟减少）。

# pytorch-lightning

https://github.com/Lightning-AI/pytorch-lightning

零代码改变便可在多 GPU 上预训练、微调和部署 AI 模型。

# onnxruntime

https://github.com/microsoft/onnxruntime

ONNX Runtime 是一个跨平台的推理引擎，支持多种深度学习框架。它集成了 Triton 以提高模型推理的速度和效率。

# pytorch

https://github.com/pytorch/pytorch

pytorch 中可以通过 `torch.compile` 使用 triton 内核。

# fauxpilot

https://github.com/fauxpilot/fauxpilot

GitHub Copilot 的本地托管替代方案。

# MOSS

[OpenMOSS](https://github.com/OpenMOSS/MOSS)

MOSS 是一个来自复旦大学的支持中英双语和多种插件的开源对话语言模型。moss-moon 系列模型具有 160 亿参数，使用 Triton 显著降低推理成本。

# QAnything

https://github.com/netease-youdao/QAnything

网易有道的 QAnything (Question and Answer based on Anything)，致力于支持任意格式文件或数据库的本地知识库问答系统。

# 其他使用

- Hugging Face 使用 Triton 与 TensorRT 将使用 BERT 模型处理推理的时间缩短至不到 1 毫秒。
- 美国运通使用 Triton 实时监测每年 1 万亿美元交易中的欺诈行为。
- 微软在 Azure 上使用 Triton 提供可扩展的高性能推理，支持 [GrammarLink 的 AI ](https://blogs.nvidia.com/blog/microsoft-triton-ai-grammar-word/)。Azure 机器学习客户可以利用 Triton 对多个框架、实时、批量和流式推理、动态批处理和并发执行的支持。GrammarLink 是其 Microsoft Word 在线编辑器，预计每年将处理多达 5 万亿次查询。
- 伦敦 Intelligent Voice 公司的 LexIQal 系统，使用 Triton 进行 AI 推理，检测保险和金融服务中的欺诈行为。
