# 基于 RAG 的私有知识库智能问答系统与服务化部署

基于 LangChain + Chroma + 通义千问（DashScope）构建的私有知识库问答系统：Streamlit 负责文档上传入库，FastAPI 提供同步与流式问答服务，检索层采用 BM25 + 向量混合检索，并使用 RAGAS 完成 30 题量化评估。

## 核心特性

- 多格式知识库入库：PDF、XLSX、TXT、CSV，基于 MD5 去重、1000 字分块写入 Chroma。
- 混合检索：jieba 分词 BM25 + 向量检索，通过 RRF 融合，解决型号、接口名、数值等精确词召回差的问题。
- 服务化问答：FastAPI 提供同步问答、SSE 流式输出、健康检查、文档列表、会话清理接口。
- 多轮会话：按 session_id 隔离，历史落盘为 JSON 文件。
- 量化评估：RAGAS 四项指标（faithfulness、answer_relevancy、context_precision、context_recall），支持分批执行与失败补跑。

## 系统架构

```
Streamlit 上传页 ──解析 / 去重 / 切块──▶ Chroma 向量库
                                            │
用户 / 客户端 ──HTTP / SSE──▶ FastAPI ──▶ 混合检索（BM25 + 向量，RRF 融合）
                                            │
                                            ▼
                                  通义千问（DashScope）
                                            │
                                  回答 + 引用来源
```

## 目录结构

```
.
├── app_fileuploader.py      # Streamlit 知识库上传页
├── app_qa.py                # Streamlit 本地问答调试页
├── main.py                  # FastAPI 问答服务
├── rag.py                   # RAG 链路：检索、提示词、多轮会话、流式输出
├── vector_stores.py         # Chroma + BM25 混合检索器
├── knowledge_base.py        # 文档切分、MD5 去重、写向量库
├── file_history_store.py    # 基于文件的会话历史存储
├── config_data.py           # 全局配置（模型、分块、检索参数）
├── start_api.bat            # Windows 一键启动 API
├── requirements.txt
├── data/                    # 原始知识库文档
├── chroma_db/               # Chroma 持久化目录
├── history/                 # 会话历史 JSON
├── md5.text                 # 已入库文档指纹
└── evaluation/
    ├── eval_dataset.csv     # 30 条中文问答评测集
    ├── run_ragas.py         # RAGAS 批量评估
    ├── check_retrieval.py   # 检索命中率诊断
    └── results/             # 评估结果 CSV
```

## 环境要求

- Python 3.10+
- 通义千问 API Key（DashScope）
- Windows / macOS / Linux 均可运行，启动脚本 `start_api.bat` 面向 Windows

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

在项目根目录创建 `.env` 文件（已加入 `.gitignore`）：

```dotenv
DASHSCOPE_API_KEY=你的 Key
```

`rag.py` 会通过 `python-dotenv` 自动读取该变量。模型名、语料路径、分块与检索参数统一在 `config_data.py` 中配置：

```python
chunk_size = 1000
chunk_overlap = 100
similarity_threshold = 5      # 向量检索 Top-K
bm25_k = 5                    # BM25 检索 Top-K
hybrid_vector_weight = 0.5    # 混合检索权重
hybrid_bm25_weight = 0.5
```

### 3. 上传知识库

```bash
streamlit run app_fileuploader.py
```

在页面中选择 PDF / XLSX / TXT / CSV 文件上传，系统会自动解析、去重、切块并写入 `chroma_db/`。重复上传同一文件会被 MD5 去重拦截。

### 4. 启动问答服务

```bash
start_api.bat
```

或手动启动：

```bash
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```

启动后访问 `http://127.0.0.1:8010/docs` 查看交互式接口文档。端口被占用时修改 `start_api.bat` 中的端口即可。

### 5. 调用示例

同步问答：

```bash
curl -X POST http://127.0.0.1:8010/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"HAK180 的电源规格是什么？\",\"session_id\":\"demo\"}"
```

流式问答（SSE）：

```bash
curl -N -X POST http://127.0.0.1:8010/api/chat/stream \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"华为显示器 B3-211H 如何关机？\",\"session_id\":\"demo\"}"
```

健康检查与文档列表：

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/api/documents
```

### 6. 本地问答调试页（可选）

```bash
streamlit run app_qa.py
```

该页面直接调用 `RagService`，仅用于本地调试；正式问答请统一走 FastAPI。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 服务状态与当前向量库 chunk 数 |
| GET | `/api/documents` | 已入库文档及各自 chunk 数 |
| POST | `/api/chat` | 同步问答，返回回答、引用来源、session_id |
| POST | `/api/chat/stream` | SSE 流式问答，事件类型：start / token / sources / error |
| DELETE | `/api/sessions/{session_id}` | 清空指定会话历史 |

请求体：

```json
{"question": "问题内容", "session_id": "可选，不传则自动生成"}
```

## 检索优化

纯向量检索对型号、接口名、数值这类精确词汇不敏感。项目内置了检索诊断脚本，用评测集逐题检查标准答案原文是否出现在返回的 chunk 中：

```bash
python evaluation/check_retrieval.py
```

诊断结果：

| 检索方式 | 命中率 |
|---|---|
| 纯向量 k=2（原方案） | 19/30 |
| 纯向量 k=5 | 24/30 |
| BM25 + 向量混合检索 | 30/30 |

优化措施（`vector_stores.py`）：使用 jieba 对文档与问题分词构建 BM25 索引，与 Chroma 向量检索组成 `EnsembleRetriever`，按 RRF 融合，两路各取 Top-5、权重 0.5/0.5。

## 评估

### 评测集

`evaluation/eval_dataset.csv` 共 30 条中文问答，覆盖笔记本、显示器、打印机手册与产品价格表，字段为 `question,ground_truth`。

### 运行评估

```bash
# 全量评估
python evaluation/run_ragas.py

# 分批评估（受大模型调用时长限制时使用）
python evaluation/run_ragas.py --offset 0 --limit 5
python evaluation/run_ragas.py --offset 5 --limit 5

# 单题补跑
python evaluation/run_ragas.py --offset 21 --limit 1
```

每题使用独立 session，调用 `RagService.answer` 获取回答与检索上下文，指标包括：

- `faithfulness`：回答是否由检索上下文支撑
- `answer_relevancy`：回答与问题的相关性
- `context_precision`：检索上下文中有用内容的占比
- `context_recall`：参考答案信息被检索上下文覆盖的比例

结果写入 `evaluation/results/ragas_<时间戳>.csv`（UTF-8-BOM，可直接用 Excel 打开）。

### 优化前后对比（前 15 题同题对比）

| 指标 | 纯向量 k=2 | 混合检索 | 变化 |
|---|---|---|---|
| faithfulness | 0.7250 | 1.0000 | +0.2750 |
| answer_relevancy | 0.5865 | 0.9877 | +0.4012 |
| context_precision | 0.6000 | 0.7674 | +0.1674 |
| context_recall | 0.5333 | 1.0000 | +0.4667 |

### 30 题整体结果（混合检索）

| 指标 | 均值 |
|---|---|
| faithfulness | 0.9762 |
| answer_relevancy | 0.9800 |
| context_precision | 0.7559 |
| context_recall | 1.0000 |

完整结果见 `evaluation/results/ragas_30_hybrid_final_*.csv`。

## 实现说明

- **分块与去重**：文本超过 1000 字时使用 `RecursiveCharacterTextSplitter` 切分；整个文件内容计算 MD5，已入库则跳过。
- **嵌入分批**：DashScope `text-embedding-v4` 单次最多 10 条，上传页对 `embed_documents` 做了每 10 条一批的封装。
- **会话历史**：`file_history_store.py` 按 session_id 写入 `history/<session_id>`，通过 `RunnableWithMessageHistory` 注入多轮上下文。
- **流式输出**：`RagService.astream_answer` 逐 token 产出，API 层转换为 SSE，结束事件附带引用来源。
- **评估稳定性**：RAGAS 使用 `RunConfig(timeout=300, max_workers=8)`、`raise_exceptions=False`，单题超时不影响整体，可单独补跑。

## 已知问题与后续计划

- `context_precision` 为 0.756，混合检索每次最多返回 10 条上下文，会混入无关片段；计划增加 reranker 或截断到 Top-6。
- 服务为单进程本地部署，未提供鉴权、限流与并发写保护。
- 评测集为 30 题单轮问答，语料为 7 份公开手册，规模有限；后续可扩充题库并引入人工标注。
