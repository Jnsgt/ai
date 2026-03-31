# Cross-Language DL Unified

一个用于 **Python / Node.js / Java** 的跨语言 ONNX 推理一致性测试工具。

本项目旨在帮助开发者将同一个 ONNX 模型分别部署到不同语言运行环境中执行推理，并自动比较输出结果，从而验证多语言部署的一致性，辅助发现输入格式、数值精度、输出结构等方面的问题。

---

## 一、项目简介

在深度学习模型实际部署中，经常会出现这样的场景：

- 使用 **Python** 完成模型开发与验证
- 使用 **Node.js** 构建服务接口或前端推理组件
- 使用 **Java** 集成到企业级后端系统中

理论上，只要使用的是同一个 ONNX 模型，并且输入完全一致，那么不同语言环境下的推理结果应当保持一致。  
但在实际部署过程中，仍然可能出现以下问题：

- 输入张量名称不一致
- 输入 shape 处理错误
- 数据类型转换不一致
- 输出节点读取错误
- 某个后端推理结果存在数值偏差
- 输出 JSON 格式不统一，难以自动比较

本项目正是为了解决这些问题而设计的。它可以自动完成以下流程：

1. 读取 ONNX 模型
2. 读取输入 JSON
3. 分别调用 Python / Node.js / Java 后端执行推理
4. 保存各后端的原始输出结果
5. 自动计算输出差异指标
6. 生成比较报告，判断不同后端是否一致

---

## 二、项目功能

本项目目前支持以下功能：

### 1. 多语言 ONNX 推理
支持使用以下运行时执行同一 ONNX 模型：

- Python + onnxruntime
- Node.js + onnxruntime-node
- Java + ONNX Runtime Java API

### 2. 自动比较推理结果
自动对不同后端的输出进行两两比较，并计算以下误差指标：

- 最大绝对误差（max absolute difference）
- 平均绝对误差（mean absolute difference）
- 最大相对误差（max relative difference）
- 均方根误差（RMSE）
- L2 距离（L2 distance）
- allclose 判定结果

### 3. 输出统一格式的 JSON 结果
不同后端均尽量输出统一结构的 JSON，便于自动分析与比较。

### 4. 提供示例模型与输入
项目内置了简单线性回归模型和多层感知机（MLP）模型，方便快速测试。

### 5. 支持批量测试与可视化
项目中还包含批量 case 比较和误差可视化相关脚本，可用于扩展实验分析。

---

## 三、项目结构

整理后的项目目录结构如下：

```text
cross-lang-dl-unified/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml
├─ package.json
├─ package-lock.json
├─ requirements.txt
├─ cldltest/
│  ├─ __init__.py
│  ├─ benchmark.py
│  ├─ cli.py
│  ├─ comparator.py
│  ├─ visualize.py
│  ├─ visualize_imported.py
│  ├─ runners/
│  │  ├─ python_runner.py
│  │  ├─ js_runner.js
│  │  ├─ python_runner_generic.py
│  │  ├─ js_runner_generic.js
│  │  ├─ run_python_onnx_legacy.py
│  │  ├─ run_node_onnx_legacy.js
│  │  └─ java_runner/
│  │     ├─ pom.xml
│  │     └─ src/main/java/OnnxJavaRunner.java
│  └─ utils/
│     └─ metrics.py
├─ docs/
│  ├─ input_format.md
│  ├─ project_structure.md
│  └─ roadmap.md
├─ examples/
│  ├─ cases/
│  │  ├─ mlp_inputs.json
│  │  └─ regression_inputs.json
│  ├─ models/
│  │  └─ linear_model.onnx
│  ├─ outputs/
│  │  ├─ python_result.json
│  │  ├─ js_result.json
│  │  └─ compare_py_js.json
│  └─ test_input.json
├─ models/
│  ├─ mlp/
│  └─ regression/
├─ scripts/
│  └─ gui_app.py
└─ tests/
   ├─ README.md
   └─ compare_outputs.py