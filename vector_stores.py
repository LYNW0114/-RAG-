from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import jieba
import config_data as config


def _tokenize(text):
    return [token for token in jieba.cut(text) if token.strip()]


class VectorStoreService():
    def __init__(self, embedding):
        """嵌入模型的传入"""
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def _get_bm25_retriever(self):
        """基于库内全部文档构建BM25关键词检索器，用于召回型号、数字等精确词"""
        data = self.vector_store.get(include=["documents", "metadatas"])
        if not data.get("documents"):
            return None
        return BM25Retriever.from_texts(
            data["documents"],
            metadatas=data.get("metadatas"),
            preprocess_func=_tokenize,
            k=config.bm25_k,
        )

    def get_vector_store(self):
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.similarity_threshold}
        )
        bm25_retriever = self._get_bm25_retriever()
        if bm25_retriever is None:
            return vector_retriever
        # 向量检索负责语义相似，BM25 负责精确词命中，用 RRF 融合两路结果
        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[config.hybrid_vector_weight, config.hybrid_bm25_weight],
        )
