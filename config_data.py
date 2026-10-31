md5_file_path = "./md5.text"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# splitter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = 1000

similarity_threshold = 5  # 向量检索返回匹配的文档数量
bm25_k = 5  # BM25 关键词检索返回的文档数量
hybrid_vector_weight = 0.5  # 混合检索中向量检索的权重
hybrid_bm25_weight = 0.5  # 混合检索中 BM25 检索的权重

embedding_model_name = "text-embedding-v4"
model_name = "qwen3.8-max"
chat_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 会话历史存储目录
chat_history_path = "./chat_history"
session_config = {"configurable": {"session_id": "user_001"}}
