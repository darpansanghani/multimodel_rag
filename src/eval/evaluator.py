import time
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from datasets import Dataset
import logging
import pandas as pd

import config as config
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper

def score(query: str, answer: str, context_chunks: list) -> dict:
    timestamp = time.time()
    res = {
        "timestamp": timestamp,
        "faithfulness": None,
        "answer_relevancy": None,
        "correctness": None
    }
    
    if not context_chunks:
        return res
        
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [context_chunks],
        "ground_truth": [""]
    }
    
    dataset = Dataset.from_dict(data)
    
    try:

        base_llm = OpenAILike(
            model=config.NVIDIA_EVAL_LLM_MODEL,
            api_key=config.NVIDIA_API_KEY,
            api_base="https://integrate.api.nvidia.com/v1",
            is_chat_model=True,
            max_retries=1
        )
        base_embeddings = HuggingFaceEmbedding(model_name=config.HF_EMBEDDING_MODEL)

        llm = LlamaIndexLLMWrapper(base_llm)
        embeddings = LlamaIndexEmbeddingsWrapper(base_embeddings)

        # In Ragas >= 0.1, answer_relevancy tries to generate multiple (n) questions by default.
        # LlamaIndex LLM wrappers don't natively support n > 1, which causes Ragas to spam
        # warnings and run them sequentially. Setting strictness=1 fixes this at the root.
        answer_relevancy.strictness = 1

        logging.getLogger("ragas").setLevel(logging.ERROR)

        eval_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, answer_correctness],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False
        )
        
        df = eval_result.to_pandas()
        if not df.empty:
            row = df.iloc[0]
            res["faithfulness"] = float(row["faithfulness"]) if "faithfulness" in row and not pd.isna(row["faithfulness"]) else None
            res["answer_relevancy"] = float(row["answer_relevancy"]) if "answer_relevancy" in row and not pd.isna(row["answer_relevancy"]) else None
            
            # Map answer_correctness to the expected 'correctness' key
            res["correctness"] = float(row["answer_correctness"]) if "answer_correctness" in row and not pd.isna(row["answer_correctness"]) else None
            
            # Ragas uses faithfulness to measure lack of hallucination. Hallucination is just the inverse.
            # f_score = res["faithfulness"]
            # res["hallucination"] = 1.0 - f_score if f_score is not None else None
        
    except Exception as e:
        print(f"[evaluator] RAGAS evaluation failed: {e}")
        
    return res
