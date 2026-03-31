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
```

---

## 四、安装环境

本项目涉及 Python、Node.js 和 Java 三套运行环境。  
如果你只想使用其中一部分后端，可以只安装对应环境。

### 1. Python 环境

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

说明：

- `requirements.txt` 用于安装 Python 运行依赖
- `pip install -e .` 用于将本项目以可编辑模式安装，安装后可直接使用 `cldltest` 命令

---

### 2. Node.js 环境

建议使用较新的 Node.js 版本。

安装依赖：

```bash
npm install
```

这一步会安装：

- `onnxruntime-node`

---

### 3. Java 环境

如果需要使用 Java 后端，请确保：

- 已安装 Java 11 或更高版本
- 已安装 Maven
- `java` 和 `mvn` 命令已加入系统环境变量 PATH

---

## 五、快速开始

### 1. 使用示例模型与输入进行测试

项目已提供一个简单线性模型和测试输入：

- 示例模型：`examples/models/linear_model.onnx`
- 示例输入：`examples/test_input.json`

输入文件内容如下：

```json
{
  "input_name": "input",
  "shape": [4, 1],
  "dtype": "float32",
  "data": [0.0, 1.0, 5.0, 10.0]
}
```

其含义为：

- 模型输入名称为 `input`
- 输入张量形状为 `[4, 1]`
- 数据类型为 `float32`
- 实际输入数据为 `[0.0, 1.0, 5.0, 10.0]`

---

### 2. 运行 Python 与 JavaScript 后端对比

在项目根目录执行：

```bash
cldltest benchmark --model examples/models/linear_model.onnx --input examples/test_input.json --outdir outputs --backends py js
```

该命令的含义如下：

- `benchmark`：执行一次跨语言 benchmark
- `--model`：指定 ONNX 模型路径
- `--input`：指定输入 JSON 路径
- `--outdir outputs`：指定结果输出目录
- `--backends py js`：表示本次运行 Python 和 JavaScript 两个后端

---

### 3. 运行 Python、JavaScript、Java 三后端对比

```bash
cldltest benchmark --model examples/models/linear_model.onnx --input examples/test_input.json --outdir outputs --backends py js java
```

---

### 4. 查看输出结果

运行成功后，会在 `outputs/` 目录下看到以下文件：

#### 各后端原始结果
- `python_result.json`
- `js_result.json`
- `java_result.json`

#### 两两比较报告
- `compare_py_js.json`
- `compare_py_java.json`
- `compare_js_java.json`

这些 JSON 文件中会记录：

- 使用的 runner 类型
- 输入信息
- 输出 shape
- 输出 data
- 数据类型
- 差异指标
- 是否通过阈值判断

---

## 六、项目运行流程

本项目的整体运行流程如下：

```text
用户提供 ONNX 模型 + 输入 JSON
        ↓
命令行入口 cli.py 解析参数
        ↓
benchmark.py 调度不同语言后端
        ↓
Python / Node.js / Java 分别执行推理
        ↓
各后端生成统一格式 JSON 输出
        ↓
comparator.py 读取结果并比较
        ↓
生成 compare_*.json 比较报告
```

从功能上看，可以将项目分成五层：

### 1. 模型层
位于 `models/` 目录，用于定义和导出测试模型。

### 2. 输入样例层
位于 `examples/` 目录，用于保存示例输入、示例模型和示例输出。

### 3. 推理执行层
位于 `cldltest/runners/`，负责调用不同语言环境执行 ONNX 推理。

### 4. 比较分析层
由 `cldltest/comparator.py` 和 `cldltest/utils/metrics.py` 组成，负责误差计算与结果判定。

### 5. 用户交互层
由 `cldltest/cli.py` 和 `scripts/gui_app.py` 提供命令行与图形界面入口。

---

## 七、核心模块说明

### 1. `cldltest/cli.py`
命令行入口文件。  
负责解析用户输入的命令行参数，并调用 benchmark 模块开始执行。

---

### 2. `cldltest/benchmark.py`
项目总调度器。  
负责：

- 创建输出目录
- 按用户选择调用不同后端
- 保存各后端输出结果
- 自动执行两两比较
- 输出最终比较报告

这是整个项目的核心调度中心。

---

### 3. `cldltest/comparator.py`
结果比较模块。  
负责读取两个结果 JSON 文件，对其中的 `output_data` 进行差异分析，并生成报告。

主要输出内容包括：

- shape 是否一致
- dtype 是否一致
- 最大绝对误差
- 平均绝对误差
- 最大相对误差
- RMSE
- 是否通过阈值判断

---

### 4. `cldltest/utils/metrics.py`
误差指标计算模块。  
提供多个常用的数值差异评价函数，包括：

- `max_abs_diff`
- `mean_abs_diff`
- `max_rel_diff`
- `rmse`
- `l2_distance`
- `allclose`

其中 `calc_metrics()` 会将这些指标统一打包成一个字典供 comparator 使用。

---

### 5. `cldltest/runners/python_runner.py`
Python 后端执行模块。

主要流程如下：

1. 读取输入 JSON
2. 使用 `numpy` 构造输入张量
3. 使用 `onnxruntime.InferenceSession` 加载模型
4. 执行推理
5. 将输出保存为统一格式的 JSON

---

### 6. `cldltest/runners/js_runner.js`
JavaScript 后端执行模块。

主要流程与 Python 类似：

1. 读取输入 JSON
2. 构造 `Float32Array`
3. 创建 ONNX Runtime Tensor
4. 加载模型
5. 执行推理
6. 写出统一格式的 JSON

需要注意的是，该版本默认使用名为 `output` 的输出节点名。如果测试模型的输出节点名称不是 `output`，则可能需要修改代码。

---

### 7. `cldltest/runners/java_runner/`
Java 后端执行模块。  
这是一个独立的 Maven 小工程，其中：

- `pom.xml`：定义 Java 依赖和运行配置
- `OnnxJavaRunner.java`：负责实际读取输入、执行推理并输出 JSON

当前 Java runner 对简单回归模型支持较好，但对复杂输出 shape 的适配仍有改进空间。

---

### 8. `cldltest/visualize_imported.py`
比较结果可视化模块。  
可将比较报告中的误差数据绘制成图，用于批量实验展示和结果分析。

---

## 八、模型与示例数据说明

### 1. `models/regression/`
该目录提供一个简单线性回归模型。

#### `simple_regression.py`
定义模型关系：

```text
y = 2x + 1
```

#### `export_to_onnx.py`
将该模型导出为 ONNX 格式。

该模型适合作为一致性测试的入门示例，因为其理论输出非常容易验证。

例如输入：

- 0 → 1
- 1 → 3
- 5 → 11
- 10 → 21

---

### 2. `models/mlp/`
该目录提供一个简单多层感知机（MLP）模型。

#### `mlp_model.py`
定义一个由全连接层和 ReLU 组成的小型神经网络。

#### `export_mlp_to_onnx.py`
将 MLP 导出为 ONNX 格式。

该模型适合测试比线性模型更复杂的输出一致性。

---

### 3. `examples/`
该目录用于提供示例数据和演示资源。

#### `examples/test_input.json`
基础测试输入。

#### `examples/cases/regression_inputs.json`
回归模型的测试输入集合。

#### `examples/cases/mlp_inputs.json`
MLP 模型的测试输入集合。

#### `examples/models/linear_model.onnx`
示例 ONNX 模型。

#### `examples/outputs/`
示例输出与比较结果，用于展示工具运行效果。

---

## 九、输入输出格式说明

### 1. 主 benchmark 输入格式

输入 JSON 格式如下：

```json
{
  "input_name": "input",
  "shape": [4, 1],
  "dtype": "float32",
  "data": [0.0, 1.0, 5.0, 10.0]
}
```

字段说明：

- `input_name`：模型输入节点名称
- `shape`：输入张量形状
- `dtype`：输入数据类型
- `data`：平铺后的输入数据

---

### 2. 主 benchmark 输出格式

各后端输出 JSON 大体包含以下字段：

```json
{
  "runner": "...",
  "model": "...",
  "input_name": "...",
  "input_shape": [...],
  "input_data": [...],
  "output_shape": [...],
  "output_data": [...],
  "dtype": "...",
  "status": "success"
}
```

这些字段为后续自动比较提供了统一基础。

---

## 十、批量测试与旧版 runner

除了主 benchmark 线路之外，项目中还保留了部分扩展或历史版本脚本，例如：

- `python_runner_generic.py`
- `js_runner_generic.js`
- `run_python_onnx_legacy.py`
- `run_node_onnx_legacy.js`

这些脚本通常用于：

- 批量 case 测试
- 兼容旧实验流程
- 对不同输入格式进行处理

与主 benchmark 相比，这部分脚本更偏实验性或辅助性。

对应的批量比较脚本为：

- `tests/compare_outputs.py`

它主要用于对批量 case 的输出结果逐条比较，并生成整体统计报告。

---

## 十一、图形界面说明

项目中提供了一个图形界面入口：

```text
scripts/gui_app.py
```

理论上可通过以下命令运行：

```bash
python scripts/gui_app.py
```

图形界面支持：

- 选择 ONNX 模型
- 选择输入 JSON
- 指定输出目录
- 勾选需要运行的后端
- 查看运行日志
- 查看结果概览

不过需要注意，当前整理后的项目中，GUI 的默认路径仍需进一步调整，因此**当前更推荐优先使用命令行 CLI 方式运行**。

---

## 十二、适用场景

本项目适用于以下场景：

- ONNX 模型跨语言部署一致性验证
- Python / Node.js / Java 推理结果对比实验
- 课程设计与实验项目展示
- 开源项目原型展示
- 多后端模型部署调试
- 测试不同运行时的精度差异

---

## 十三、当前优点与后续改进方向

### 当前优点
- 项目结构较清晰
- 支持 Python / JavaScript / Java 三种后端
- 输出结果统一，便于比较
- 自带示例模型和测试输入
- 已具备开源项目基础框架

### 当前仍可改进的方向
- 修正 GUI 默认路径问题
- 提升 Java runner 对复杂输出 shape 的通用性
- 提升 JavaScript runner 对输出节点名的自动适配能力
- 增强批量测试的 CLI 支持
- 增加更多模型类型与示例数据
- 完善结果可视化模块

---

## 十四、常见问题

### 1. 为什么 Python 和 JavaScript 输出有微小差异？
不同语言运行时在底层数值实现上可能存在极小浮点误差，因此需要通过误差阈值而不是逐位完全相等来判断一致性。

### 2. 为什么 Java 后端运行失败？
请确认：

- Java 已安装
- Maven 已安装
- `java` 和 `mvn` 命令可在终端直接使用
- ONNX Runtime Java 依赖可正常下载

### 3. 为什么 JavaScript 后端找不到输出？
当前 `js_runner.js` 默认读取名为 `output` 的输出节点。如果模型输出节点名称不同，需要修改相关代码逻辑。

### 4. 为什么推荐优先使用 CLI？
当前项目中的 GUI 路径在整理后尚需同步修复，因此命令行方式更稳定、可控、便于排查问题。

---

## 十五、许可证

本项目采用仓库根目录下的 `LICENSE` 文件所指定的开源许可证。

---

## 十六、总结

本项目是一个围绕 **ONNX 跨语言部署一致性验证** 所构建的工具原型。  
它将模型、输入、跨语言运行、结果保存、误差比较和可视化分析整合到同一套工程结构中，适合作为：

- 课程项目
- 实验工具
- 开源原型
- 竞赛展示基础框架

如果你希望进一步扩展，本项目还可以继续朝以下方向发展：

- 支持更多语言后端（如 C++）
- 支持更多模型类型
- 支持批量 benchmark 命令
- 支持 Web 可视化界面
- 支持自动生成 HTML/PDF 报告

---
