# RAG Evaluation

Evaluating Retrieval-Augmented Generation systems -- and Graph RAG systems in particular -- requires measuring both retrieval quality and generation quality. This section covers frameworks, metrics, and practical approaches for rigorous RAG evaluation.

> **LangChain Evaluation**: https://python.langchain.com/docs/how_to/#evaluation

## Why RAG Evaluation Is Hard

Traditional NLP evaluation (BLEU, ROUGE) measures surface similarity. RAG evaluation must assess:

1. **Did the retriever find the right context?** (retrieval quality)
2. **Did the LLM use the context correctly?** (faithfulness)
3. **Is the answer actually correct?** (correctness)
4. **Is the answer relevant to the question?** (relevancy)

For Graph RAG specifically, we also need to assess:
5. **Did the graph traversal reach relevant subgraphs?**
6. **Did community summaries provide useful global context?**
7. **How does graph RAG compare to vector RAG on the same questions?**

## RAGAS Framework

RAGAS (Retrieval-Augmented Generation Assessment) is the most widely adopted framework for RAG evaluation.

> **RAGAS Documentation**: https://docs.ragas.io/

### Core Metrics

| Metric | What It Measures | Range | Needs Ground Truth? |
|--------|-----------------|-------|-------------------|
| **Faithfulness** | Is the answer grounded in the retrieved context? | 0-1 | No |
| **Answer Relevancy** | Does the answer address the question? | 0-1 | No |
| **Context Precision** | Are the retrieved contexts relevant? | 0-1 | Yes (ground truth answer) |
| **Context Recall** | Do retrieved contexts contain all needed info? | 0-1 | Yes (ground truth answer) |
| **Answer Correctness** | Is the answer factually correct? | 0-1 | Yes (ground truth answer) |

### Setting Up RAGAS

```python
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "Who founded Tesla?",
        "What is the capital of France?",
    ],
    "answer": [
        "Tesla was founded by Elon Musk, JB Straubel, Martin Eberhard, Marc Tarpenning, and Ian Wright.",
        "The capital of France is Paris.",
    ],
    "contexts": [
        ["Tesla, Inc. was founded in 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined as chairman and later became CEO."],
        ["Paris is the capital and largest city of France, situated on the Seine river."],
    ],
    "ground_truth": [
        "Tesla was founded by Martin Eberhard and Marc Tarpenning in 2003. Elon Musk was an early investor and chairman.",
        "Paris is the capital of France.",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
)

print(results)
# {'faithfulness': 0.85, 'answer_relevancy': 0.92, 'context_precision': 0.90, ...}
```

### Understanding Each Metric

#### Faithfulness

Measures whether every claim in the answer can be traced back to the retrieved context. An answer that adds information not in the context scores low.

```
Context: "Marie Curie won the Nobel Prize in Physics in 1903."
Answer: "Marie Curie won the Nobel Prize in Physics in 1903 and Chemistry in 1911."
Faithfulness: 0.5 (the Chemistry claim is not in the context)
```

#### Answer Relevancy

Measures whether the answer addresses the question asked. An answer that is factually correct but off-topic scores low.

```
Question: "What is the capital of France?"
Answer: "France is a country in Western Europe with a population of 67 million."
Relevancy: 0.2 (correct facts, but doesn't answer the question)
```

#### Context Precision

Measures how many of the retrieved context chunks are actually relevant to answering the question. High precision means less noise in retrieval.

#### Context Recall

Measures whether the retrieved contexts contain all the information needed to answer the question. High recall means no missing information.

## LLM-as-Judge Evaluation

When you do not have ground truth answers, use an LLM to evaluate response quality.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert evaluator. Score the following answer on a scale of 1-5 for each criterion.

Criteria:
1. Correctness: Is the answer factually accurate?
2. Completeness: Does the answer cover all aspects of the question?
3. Conciseness: Is the answer appropriately concise without unnecessary information?
4. Groundedness: Is the answer grounded in the provided context (no hallucination)?

Return your scores as JSON: {{"correctness": N, "completeness": N, "conciseness": N, "groundedness": N, "reasoning": "..."}}"""),
    ("human", """Question: {question}
Context: {context}
Answer: {answer}

Evaluate this answer.""")
])

judge_chain = judge_prompt | judge_llm.with_structured_output(dict)
```

> **LangChain structured output**: https://python.langchain.com/docs/how_to/structured_output/

## Comparing Vector RAG vs Graph RAG

A key question: does your Graph RAG system actually outperform simple vector RAG? Design a controlled comparison.

### Experiment Design

```python
from dataclasses import dataclass

@dataclass
class RAGComparison:
    question: str
    vector_rag_answer: str
    graph_rag_answer: str
    ground_truth: str
    vector_rag_contexts: list[str]
    graph_rag_contexts: list[str]

def run_comparison(
    questions: list[str],
    ground_truths: list[str],
    vector_rag_chain,
    graph_rag_chain
) -> list[RAGComparison]:
    """Run the same questions through both RAG systems."""
    comparisons = []
    for q, gt in zip(questions, ground_truths):
        v_result = vector_rag_chain.invoke({"query": q})
        g_result = graph_rag_chain.invoke({"query": q})

        comparisons.append(RAGComparison(
            question=q,
            vector_rag_answer=v_result["result"],
            graph_rag_answer=g_result["result"],
            ground_truth=gt,
            vector_rag_contexts=v_result.get("source_documents", []),
            graph_rag_contexts=g_result.get("source_documents", []),
        ))
    return comparisons
```

### Question Categories Where Graph RAG Excels

| Question Type | Vector RAG | Graph RAG | Example |
|--------------|-----------|-----------|---------|
| Multi-hop reasoning | Poor | Strong | "Who mentored the person who founded X?" |
| Aggregation / global | Poor | Strong | "What are the main themes across all documents?" |
| Entity-centric | Moderate | Strong | "Tell me everything about entity X" |
| Relationship queries | Poor | Strong | "How are X and Y connected?" |
| Single-fact lookup | Strong | Moderate | "When was X founded?" |
| Similarity search | Strong | Moderate | "Find documents similar to X" |

### Statistical Significance

Do not rely on averages alone. Use paired statistical tests:

```python
from scipy import stats

def compare_systems(vector_scores: list[float], graph_scores: list[float]) -> dict:
    """Compare two RAG systems with statistical testing."""
    # Paired t-test (same questions, different systems)
    t_stat, p_value = stats.ttest_rel(graph_scores, vector_scores)

    return {
        "vector_mean": sum(vector_scores) / len(vector_scores),
        "graph_mean": sum(graph_scores) / len(graph_scores),
        "improvement": (sum(graph_scores) - sum(vector_scores)) / len(vector_scores),
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
```

## A/B Testing in Production

For production Graph RAG systems, run A/B tests with real users.

### Key Metrics to Track

```python
@dataclass
class ABTestMetrics:
    # Retrieval metrics
    latency_p50_ms: float
    latency_p95_ms: float

    # Quality metrics (from user feedback or LLM judge)
    thumbs_up_rate: float        # User satisfaction
    answer_length_avg: float     # Verbosity
    hallucination_rate: float    # From faithfulness checks

    # Graph-specific metrics
    subgraph_size_avg: float     # Nodes retrieved per query
    traversal_depth_avg: float   # Hops in graph traversal
    community_summaries_used: float  # For Microsoft GraphRAG
```

### Logging for Evaluation

```python
import json
import time

def log_rag_interaction(
    question: str,
    answer: str,
    contexts: list[str],
    system: str,  # "vector" or "graph"
    metadata: dict = None
):
    """Log RAG interactions for later evaluation."""
    log_entry = {
        "timestamp": time.time(),
        "system": system,
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "num_contexts": len(contexts),
        "answer_length": len(answer),
        "metadata": metadata or {},
    }
    with open("rag_evaluation_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

## Building an Evaluation Dataset

A good evaluation dataset for Graph RAG should include:

1. **Multi-hop questions** (require traversing multiple relationships)
2. **Aggregation questions** (require global graph understanding)
3. **Entity-centric questions** (everything about one entity)
4. **Comparison questions** (compare two entities)
5. **Temporal questions** (changes over time)

```python
eval_questions = [
    # Multi-hop
    {"question": "Who are the advisors of the CEO of companies that use TensorFlow?",
     "type": "multi_hop", "expected_hops": 3},

    # Aggregation
    {"question": "What are the most common research topics across all papers in 2024?",
     "type": "aggregation", "expected_hops": 0},

    # Entity-centric
    {"question": "Summarize everything we know about Marie Curie from the knowledge graph",
     "type": "entity_centric", "expected_hops": 1},

    # Comparison
    {"question": "Compare the research output of MIT and Stanford in NLP",
     "type": "comparison", "expected_hops": 2},

    # Temporal
    {"question": "How has the collaboration network changed between 2020 and 2024?",
     "type": "temporal", "expected_hops": 2},
]
```

## Automated Evaluation Pipeline

```python
from langchain_openai import ChatOpenAI

async def evaluate_graph_rag_pipeline(
    rag_chain,
    eval_dataset: list[dict],
    judge_llm=None,
) -> dict:
    """Run automated evaluation of a Graph RAG pipeline."""
    if judge_llm is None:
        judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

    results = []
    for item in eval_dataset:
        # Get RAG answer
        response = rag_chain.invoke({"query": item["question"]})

        # Judge the response
        judgment = judge_chain.invoke({
            "question": item["question"],
            "context": str(response.get("source_documents", [])),
            "answer": response["result"],
        })

        results.append({
            "question": item["question"],
            "type": item.get("type"),
            "scores": judgment,
        })

    # Aggregate by question type
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r["scores"])

    summary = {}
    for qtype, scores in by_type.items():
        summary[qtype] = {
            metric: sum(s[metric] for s in scores) / len(scores)
            for metric in scores[0].keys()
            if isinstance(scores[0][metric], (int, float))
        }

    return {"per_question": results, "by_type": summary}
```

> **LangChain async**: https://python.langchain.com/docs/how_to/batch/

## Best Practices

1. **Use RAGAS for standardized metrics** when you have ground truth
2. **Use LLM-as-judge** when ground truth is unavailable
3. **Always compare against a vector RAG baseline** to justify graph complexity
4. **Test across question types** -- Graph RAG shines on multi-hop, not single-fact
5. **Use statistical tests** (paired t-test) for reliable comparisons
6. **Log everything** for post-hoc analysis
7. **Evaluate retrieval and generation separately** to diagnose issues

## Next Steps

- [01 - Graph Quality Metrics](./01-graph-quality-metrics.md) -- evaluate the underlying knowledge graph
- [Hybrid Retrieval](../05-advanced-topics/02-hybrid-retrieval.md) -- combine vector and graph for better results
