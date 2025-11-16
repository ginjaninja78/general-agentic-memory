# general-agentic-memory
A general memory system for agents, powered by deep-research


<h5 align="center"> 🎉 If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

**General Agentic Memory (GAM)** provides a next-generation memory framework for AI agents, combining long-term retention with dynamic reasoning. Following the Just-in-Time (JIT) principle, it preserves full contextual fidelity offline while performing deep research online to build adaptive, high-utility context. With its dual-agent architecture—Memorizer and Researcher—GAM integrates structured memory with iterative retrieval and reflection, achieving state-of-the-art performance across LoCoMo, HotpotQA, LongBench v2, and LongCodeBench benchmarks.

- **Paper**: 
- **Website**: 
- **Documentation**: 
- **YouTube Video**: 

<span id='features'/>

## ✨Key Features

* 🧠 Just-in-Time (JIT) Memory Optimization
</br> Unlike conventional Ahead-of-Time (AOT) systems, GAM performs intensive Memory Deep Research at runtime, dynamically retrieving and synthesizing high-utility context to meet real-time agent needs.

* 🔍 Dual-Agent Architecture: Memorizer & Researcher
</br> A cooperative framework where the Memorizer constructs structured memory from raw sessions, and the Researcher performs iterative retrieval, reflection, and summarization to deliver precise, adaptive context.

* 🚀 Superior Performance Across Benchmarks
</br> Achieves state-of-the-art results on LoCoMo, HotpotQA, LongBench v2, and LongCodeBench, surpassing prior systems such as A-MEM, Mem0, and MemoryOS in both F1 and BLEU-1 metrics.

* 🧩 Modular & Extensible Design
</br> Built to support flexible plug-ins for memory construction, retrieval strategies, and reasoning tools—facilitating easy integration into multi-agent frameworks or standalone LLM deployments.

* 🌐 Cross-Model Compatibility
</br> Compatible with leading LLMs such as GPT-5, GPT-4o-mini, and Qwen2.5, supporting both cloud-based and local deployments for research or production environments.

<span id='news'/>

## 📣 Latest News


## 📑 Table of Contents

* <a href='#features'>✨ Features</a>
* <a href='#news'>🔥 News</a>
* <a href='#structure'> 📁Project Structure</a>
* <a href='#pypi-mode'>🎯 Quick Start</a>
* <a href='#todo'>☑️ Todo List</a>
* <a href='#reproduce'>🔬 How to Reproduce the Results in the Paper </a>
* <a href='#doc'>📖 Documentation </a>
* <a href='#cite'>🌟 Cite</a>
* <a href='#community'>🤝 Join the Community</a>




<span id='structure'/>

## 🏗️	System Architecture
![logo](./assets/GAM-memory.png)



## 🏗️ Project Structure

```
general-agentic-memory/
├── gam/                          # 核心 GAM 包
│   ├── __init__.py              # 包初始化文件
│   ├── agents/                  # 智能代理实现
│   │   ├── memory_agent.py     # MemoryAgent - 记忆构建
│   │   └── research_agent.py   # ResearchAgent - 深度研究
│   ├── generator/               # LLM 生成器
│   │   ├── openai_generator.py # OpenAI API 生成器
│   │   └── vllm_generator.py   # VLLM 本地生成器
│   ├── retriever/               # 检索器
│   │   ├── index_retriever.py  # 索引检索
│   │   ├── bm25.py             # BM25 关键词检索
│   │   └── dense_retriever.py  # Dense 语义检索
│   ├── prompts/                 # 提示词模板
│   ├── schemas/                 # 数据模型
│   └── config/                  # 配置管理
├── benchmarks/                  # 🆕 评估基准套件
│   ├── __init__.py
│   ├── run.py                  # CLI 统一入口
│   ├── README.md               # 评估文档
│   ├── MIGRATION.md            # 迁移指南
│   ├── datasets/               # 数据集适配器
│   │   ├── base.py            # 评估基类
│   │   ├── hotpotqa.py        # HotpotQA 多跳问答
│   │   ├── narrativeqa.py     # NarrativeQA 叙事问答
│   │   ├── locomo.py          # LoCoMo 对话记忆
│   │   └── ruler.py           # RULER 长上下文评估
│   └── utils/                  # 评估工具
│       ├── chunking.py        # 文本切分
│       └── metrics.py         # 评估指标
├── scripts/                     # 🆕 Shell 脚本
│   ├── eval_hotpotqa.sh
│   ├── eval_narrativeqa.sh
│   ├── eval_locomo.sh
│   ├── eval_ruler.sh
│   └── eval_all.sh
├── examples/                     # 使用示例
│   └── quickstart/              # 快速开始示例
│       ├── basic_usage.py       # 基础使用示例
│       └── model_usage.py       # 模型选择示例
├── assets/                      # 资源文件
│   └── GAM-memory.png
├── docs/                        # 文档目录
├── setup.py                     # 安装配置
├── pyproject.toml              # 现代项目配置
├── requirements.txt             # 依赖列表
└── README.md                   # 项目说明
```


<span id='pypi-mode'/>

## 📖GAM Getting Started

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/general-agentic-memory.git
cd general-agentic-memory

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 💡 Quick Start

```python
from gam import MemoryAgent, OpenRouterModel, build_session_chunks_from_text

# Initialize LLM
llm = OpenRouterModel(
    model="gpt-4o-mini",
    api_key="your-api-key"
)

# Create memory agent
memory_agent = MemoryAgent(llm)

# Process long text
long_text = "Your long document content here..."
sessions = build_session_chunks_from_text(long_text, max_tokens=2000)

# Build memory
memory_agent.run_memory_agent(sessions=sessions)
final_memory = memory_agent.get_memory_with_abstracts()

print("Memory Events:", final_memory['events'])
print("Overall Abstract:", final_memory['abstract'])
```

### 📚 Complete Examples

For detailed examples and advanced usage, check out:
- [`examples/quickstart/basic_usage.py`](./examples/quickstart/basic_usage.py) - Complete workflow examples with long text processing and deep research
- [`examples/quickstart/model_usage.py`](./examples/quickstart/model_usage.py) - Model selection and configuration examples


<span id='todo'/>

## ☑️ Todo List


Have ideas or suggestions? Contributions are welcome! Please feel free to submit issues or pull requests! 🚀

<span id='reproduce'/>

## 🔬 How to Reproduce the Results in the Paper

我们提供了完整的评估框架来复现论文中的实验结果。

### 快速开始

```bash
# 1. 准备数据集
mkdir -p data
# 将数据集放入 data/ 目录

# 2. 设置环境变量
export OPENAI_API_KEY="your_api_key_here"

# 3. 运行评估
# HotpotQA
bash scripts/eval_hotpotqa.sh --data-path data/hotpotqa.json

# NarrativeQA
bash scripts/eval_narrativeqa.sh --data-path narrativeqa --max-samples 100

# LoCoMo
bash scripts/eval_locomo.sh --data-path data/locomo.json

# RULER
bash scripts/eval_ruler.sh --data-path data/ruler.jsonl --dataset-name niah_single_1

# 或运行所有评估
bash scripts/eval_all.sh
```

### 使用 Python CLI

```bash
python -m benchmarks.run \
    --dataset hotpotqa \
    --data-path data/hotpotqa.json \
    --generator openai \
    --model gpt-4 \
    --retriever dense \
    --max-samples 100
```

### 详细文档

完整的评估文档请查看：
- [benchmarks/README.md](./benchmarks/README.md) - 评估框架使用指南
- [benchmarks/MIGRATION.md](./benchmarks/MIGRATION.md) - 从旧版本迁移指南

### 支持的数据集

| 数据集 | 任务类型 | 评估指标 | 文档 |
|--------|----------|----------|------|
| **HotpotQA** | 多跳问答 | EM, F1 | [查看](./benchmarks/datasets/hotpotqa.py) |
| **NarrativeQA** | 叙事问答 | F1, ROUGE-L | [查看](./benchmarks/datasets/narrativeqa.py) |
| **LoCoMo** | 对话记忆 | EM, F1 | [查看](./benchmarks/datasets/locomo.py) |
| **RULER** | 长上下文 | Accuracy | [查看](./benchmarks/datasets/ruler.py) |

<span id='doc'/>

## 📖 Documentation

A more detailed documentation is coming soon 🚀, and we will update in the Documentation page.

<span id='cite'/>

## 📣 Citation
**If you find this project useful, please consider citing our paper:**



<span id='related'/>



<span id='community'/>

## 🎯 Contact us


## 🌟 Star History



## Disclaimer
