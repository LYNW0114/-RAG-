"""检索命中率诊断：对比纯向量检索与混合检索在评测集上的召回情况。

用法：
    python evaluation/check_retrieval.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from langchain_community.embeddings import DashScopeEmbeddings  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain.retrievers import EnsembleRetriever  # noqa: E402

import config_data as config  # noqa: E402
from vector_stores import VectorStoreService, _tokenize  # noqa: E402

# 每题用于判定「召回到原文」的关键词，命中任意一个即算命中
KEYWORDS = {
    1: ["充满"],
    2: ["触摸板中间", "触摸板"],
    3: ["6*2.5"],
    4: ["1.5V", "1.5 V"],
    5: ["Thunderbolt"],
    6: ["长按电源键", "2 秒", "2秒"],
    7: ["白色闪烁"],
    8: ["短按任意功能键"],
    9: ["ECO"],
    10: ["向上短按"],
    11: ["-10℃", "45℃"],
    12: ["340W", "340 W"],
    13: ["90 g", "350 g"],
    14: ["M4 Pro"],
    15: ["A17 Pro"],
    16: ["排水孔"],
    17: ["转接线"],
    18: ["正在充电"],
    19: ["microSD"],
    20: ["网线"],
    21: ["VGA"],
    22: ["信息"],
    23: ["左上角", "右上角"],
    24: ["DP 线缆", "DP线缆", "DP"],
    25: ["15 厘米", "15厘米"],
    26: ["20 个字符", "20个字符"],
    27: ["220 V", "220V"],
    28: ["16.3"],
    29: ["80,000", "80000"],
    30: ["8999", "8,999"],
}


def hit(docs, index):
    keywords = KEYWORDS[index]
    return any(keyword in doc.page_content for doc in docs for keyword in keywords)


def main():
    dataset = pd.read_csv(EVAL_DIR / "eval_dataset.csv")
    store = VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name)).vector_store

    vector_k2 = store.as_retriever(search_kwargs={"k": 2})
    vector_k5 = store.as_retriever(search_kwargs={"k": 5})
    data = store.get(include=["documents", "metadatas"])
    bm25 = BM25Retriever.from_texts(
        data["documents"],
        metadatas=data["metadatas"],
        preprocess_func=_tokenize,
        k=config.bm25_k,
    )
    hybrid = EnsembleRetriever(
        retrievers=[vector_k5, bm25],
        weights=[config.hybrid_vector_weight, config.hybrid_bm25_weight],
    )

    retrievers = {"向量k=2(原基线)": vector_k2, "向量k=5": vector_k5, "混合检索": hybrid}
    misses = {name: [] for name in retrievers}

    for index, row in dataset.iterrows():
        question = str(row["question"])
        line = f"Q{index + 1:02d}"
        for name, retriever in retrievers.items():
            ok = hit(retriever.invoke(question), index + 1)
            line += f"  {name}:{'命中' if ok else '未命中'}"
            if not ok:
                misses[name].append(index + 1)
        print(line)

    print("\n=== 汇总 ===")
    total = len(dataset)
    for name, miss_list in misses.items():
        print(f"{name}: 命中 {total - len(miss_list)}/{total}，未命中题号 {miss_list}")


if __name__ == "__main__":
    main()
