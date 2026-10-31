"""RAGAS 批量评估脚本。

用法：
    python evaluation/run_ragas.py
    python evaluation/run_ragas.py --limit 5   # 只评估前 5 条，便于快速验证

结果保存到 evaluation/results/ragas_<时间戳>.csv
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402

from file_history_store import get_history  # noqa: E402
from rag import RagService  # noqa: E402

DATASET_PATH = EVAL_DIR / "eval_dataset.csv"
RESULTS_DIR = EVAL_DIR / "results"


def build_samples(rag: RagService, limit: int | None = None, offset: int = 0):
    dataset = pd.read_csv(DATASET_PATH)
    if offset:
        dataset = dataset.iloc[offset:]
    if limit:
        dataset = dataset.head(limit)

    samples = []
    for index, row in dataset.iterrows():
        session_id = f"ragas_eval_{index}"
        get_history(session_id).clear()
        result = rag.answer(str(row["question"]), session_id)
        samples.append(SingleTurnSample(
            user_input=str(row["question"]),
            retrieved_contexts=result["contexts"],
            response=result["answer"],
            reference=str(row["ground_truth"]),
        ))
        print(f"[{len(samples)}/{len(dataset)}] 已生成回答：{row['question']}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估私有知识库问答")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条")
    parser.add_argument("--offset", type=int, default=0, help="从第 N 条开始评估")
    args = parser.parse_args()

    rag = RagService()
    samples = build_samples(rag, args.limit, args.offset)
    evaluator_llm = LangchainLLMWrapper(rag.chat_model)
    evaluator_embeddings = LangchainEmbeddingsWrapper(rag.vector_service.embedding)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    print(f"\n开始评估 {len(samples)} 条样本，调用大模型可能需要几分钟...\n")
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        run_config=RunConfig(timeout=300, max_workers=8),
        raise_exceptions=False,
    )

    df = result.to_pandas()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"ragas_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n=== RAGAS 指标均值 ===")
    for metric in metrics:
        if metric.name in df.columns:
            print(f"{metric.name}: {df[metric.name].mean():.4f}")
    print(f"\n评估明细已保存：{output_path}")


if __name__ == "__main__":
    main()
