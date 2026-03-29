# Benchmark Datasets

Standard benchmark datasets are essential for evaluating knowledge graph systems -- from link prediction and entity alignment to multi-hop question answering and Graph RAG. This section covers the most widely used benchmarks, their characteristics, and how to load them.

## Link Prediction Benchmarks

Link prediction is the task of predicting missing edges in a knowledge graph. These benchmarks evaluate KG embedding methods (TransE, DistMult, ComplEx, RotatE, etc.).

### FB15k-237

**Source**: Derived from Freebase
**Size**: 14,541 entities, 237 relation types, 310,116 triples
**Task**: Link prediction (predict missing head or tail entity)
**Why "-237"**: The original FB15k had test leakage (inverse relations). FB15k-237 removes these.

| Split | Triples |
|-------|---------|
| Train | 272,115 |
| Valid | 17,535 |
| Test | 20,466 |

```python
# Load with PyKEEN
from pykeen.datasets import FB15k237

dataset = FB15k237()
print(f"Entities: {dataset.num_entities}")
print(f"Relations: {dataset.num_relations}")
print(f"Training triples: {dataset.training.num_triples}")
print(f"Validation triples: {dataset.validation.num_triples}")
print(f"Test triples: {dataset.testing.num_triples}")

# Train a model
from pykeen.pipeline import pipeline

result = pipeline(
    dataset="FB15k-237",
    model="TransE",
    training_kwargs=dict(num_epochs=100),
    evaluation_kwargs=dict(batch_size=256),
)
print(f"MRR: {result.metric_results.get_metric('mean_reciprocal_rank'):.4f}")
print(f"Hits@10: {result.metric_results.get_metric('hits_at_10'):.4f}")
```

```python
# Load with torchkge
from torchkge.data_structures import KnowledgeGraph
from torchkge.utils.datasets import load_fb15k237

kg_train, kg_val, kg_test = load_fb15k237()
print(f"Training: {kg_train.n_facts} facts")
```

### WN18RR

**Source**: Derived from WordNet
**Size**: 40,943 entities, 11 relation types, 93,003 triples
**Task**: Link prediction
**Why "RR"**: Reduced and Reciprocal-free version of WN18 (removes test leakage)

| Split | Triples |
|-------|---------|
| Train | 86,835 |
| Valid | 3,034 |
| Test | 3,134 |

```python
from pykeen.datasets import WN18RR

dataset = WN18RR()
print(f"Entities: {dataset.num_entities}")
print(f"Relations: {dataset.num_relations}")

# The 11 relations include: _hypernym, _derivationally_related_form,
# _instance_hypernym, _also_see, _member_meronym, _synset_domain_topic_of,
# _has_part, _member_of_domain_usage, _member_of_domain_region,
# _verb_group, _similar_to
```

### YAGO3-10

**Source**: Derived from YAGO3
**Size**: 123,182 entities, 37 relation types, 1,089,040 triples
**Task**: Link prediction
**Note**: Larger than FB15k-237, includes only entities with at least 10 relations

| Split | Triples |
|-------|---------|
| Train | 1,079,040 |
| Valid | 5,000 |
| Test | 5,000 |

```python
from pykeen.datasets import YAGO310

dataset = YAGO310()
print(f"Entities: {dataset.num_entities}")
print(f"Relations: {dataset.num_relations}")
```

### Link Prediction Metrics

| Metric | Description | Higher is Better? |
|--------|-------------|------------------|
| **MRR** (Mean Reciprocal Rank) | Average of 1/rank for correct predictions | Yes |
| **Hits@1** | % of correct entities ranked first | Yes |
| **Hits@3** | % of correct entities in top 3 | Yes |
| **Hits@10** | % of correct entities in top 10 | Yes |
| **MR** (Mean Rank) | Average rank of correct predictions | No |

## Multi-Hop Question Answering Benchmarks

These benchmarks evaluate systems that must reason over multiple pieces of evidence to answer questions -- the core capability of Graph RAG.

### HotpotQA

**URL**: https://hotpotqa.github.io/
**Size**: 113K question-answer pairs
**Task**: Multi-hop QA with supporting facts
**Hops**: 2 (each question requires combining 2 Wikipedia paragraphs)
**Key Feature**: Provides ground-truth supporting facts for evaluation

```python
from datasets import load_dataset

hotpotqa = load_dataset("hotpotqa", "fullwiki")
print(f"Train: {len(hotpotqa['train'])} questions")
print(f"Validation: {len(hotpotqa['validation'])} questions")

# Example question
example = hotpotqa["validation"][0]
print(f"Question: {example['question']}")
print(f"Answer: {example['answer']}")
print(f"Type: {example['type']}")  # "bridge" or "comparison"
print(f"Supporting facts: {example['supporting_facts']}")
```

### MuSiQue (Multi-Step Question Understanding)

**URL**: https://github.com/StonyBrookNLP/musique
**Size**: 25K questions
**Task**: Multi-hop QA (2-4 hops)
**Key Feature**: Decomposed sub-questions provided, harder than HotpotQA

```python
from datasets import load_dataset

musique = load_dataset("StonyBrookNLP/musique")
example = musique["validation"][0]
print(f"Question: {example['question']}")
print(f"Answer: {example['answer']}")
print(f"Number of hops: {example['num_hops']}")
# Also includes decomposed sub-questions
```

### Natural Questions (NQ)

**URL**: https://ai.google.com/research/NaturalQuestions
**Size**: 307K training, 7.8K dev questions
**Task**: Single-hop QA from Wikipedia
**Use**: Baseline for comparing multi-hop vs single-hop performance

```python
from datasets import load_dataset

nq = load_dataset("natural_questions", "default")
# Note: this is a large dataset (~42GB)
# Use streaming for exploration:
nq_stream = load_dataset("natural_questions", "default", streaming=True)
example = next(iter(nq_stream["train"]))
print(f"Question: {example['question']['text']}")
```

### Multi-Hop QA Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | % of predictions that exactly match the ground truth |
| **F1** | Token-level F1 between prediction and ground truth |
| **Supporting Fact EM** | % of correctly identified supporting facts |
| **Supporting Fact F1** | Token-level F1 for supporting fact identification |
| **Joint EM** | Both answer AND supporting facts must be exactly correct |

## Graph RAG Benchmarks

### GraphRAG Benchmark (2025)

As Graph RAG matures, dedicated benchmarks are emerging. Key evaluation dimensions:

| Dimension | Questions | What It Tests |
|-----------|-----------|---------------|
| Local retrieval | Entity-specific questions | Subgraph extraction quality |
| Global retrieval | Theme/summary questions | Community detection + summarization |
| Multi-hop | Chain-of-reasoning questions | Graph traversal effectiveness |
| Comparison | A vs B questions | Cross-entity relationship retrieval |

### Building Your Own Graph RAG Benchmark

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class BenchmarkQuestion(BaseModel):
    question: str
    answer: str
    question_type: str = Field(description="One of: local, global, multi_hop, comparison")
    required_entities: list[str]
    required_hops: int

def generate_benchmark_from_graph(graph_data, num_questions: int = 50):
    """Use an LLM to generate evaluation questions from a knowledge graph."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    structured_llm = llm.with_structured_output(BenchmarkQuestion)

    # Generate questions for each type
    questions = []
    for q_type in ["local", "global", "multi_hop", "comparison"]:
        prompt = f"""Given this knowledge graph data, generate a {q_type} question
        that requires understanding the graph to answer correctly.

        Graph data (sample):
        {graph_data[:3000]}

        Question type: {q_type}
        Generate a question and its correct answer."""

        for _ in range(num_questions // 4):
            q = structured_llm.invoke(prompt)
            questions.append(q)

    return questions
```

> **LangChain structured output**: https://python.langchain.com/docs/how_to/structured_output/

## Entity Alignment Benchmarks

For evaluating systems that align entities across different knowledge graphs.

### DBP15K

**Task**: Cross-lingual entity alignment (match entities between different language DBpedia editions)
**Pairs**: ZH-EN, JA-EN, FR-EN (15,000 aligned entity pairs each)

### OpenEA

**URL**: https://github.com/nju-websoft/OpenEA
**Task**: Entity alignment across KGs
**Datasets**: D-W-15K (DBpedia-Wikidata), D-Y-15K (DBpedia-YAGO)

## Dataset Loading Reference

### Using Hugging Face Datasets

```python
from datasets import load_dataset

# Most QA benchmarks are available on Hugging Face
hotpotqa = load_dataset("hotpotqa", "fullwiki")
musique = load_dataset("StonyBrookNLP/musique")
nq = load_dataset("natural_questions", "default", streaming=True)
```

### Using PyKEEN (Link Prediction)

```python
# pip install pykeen
from pykeen.datasets import FB15k237, WN18RR, YAGO310

# All datasets auto-download on first use
for DatasetClass in [FB15k237, WN18RR, YAGO310]:
    ds = DatasetClass()
    print(f"{ds.__class__.__name__}: {ds.num_entities} entities, {ds.num_relations} relations")
```

### Manual Download

```bash
# FB15k-237
wget https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/train.txt
wget https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/valid.txt
wget https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/test.txt

# WN18RR
wget https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/train.txt
wget https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/valid.txt
wget https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/test.txt
```

## Summary Table

| Dataset | Task | Size | Hops | Key Metric | Access |
|---------|------|------|------|-----------|--------|
| FB15k-237 | Link prediction | 310K triples | N/A | MRR, Hits@10 | PyKEEN / download |
| WN18RR | Link prediction | 93K triples | N/A | MRR, Hits@10 | PyKEEN / download |
| YAGO3-10 | Link prediction | 1.1M triples | N/A | MRR, Hits@10 | PyKEEN / download |
| HotpotQA | Multi-hop QA | 113K questions | 2 | EM, F1 | Hugging Face |
| MuSiQue | Multi-hop QA | 25K questions | 2-4 | EM, F1 | Hugging Face |
| Natural Questions | Single-hop QA | 307K questions | 1 | EM, F1 | Hugging Face |
| DBP15K | Entity alignment | 15K pairs | N/A | Hits@1 | GitHub |

## Next Steps

- [01 - Public Knowledge Graphs](./01-public-knowledge-graphs.md) -- use these KGs as data sources
- [Graph Quality Metrics](../09-evaluation-metrics/01-graph-quality-metrics.md) -- evaluate your graph
- [RAG Evaluation](../09-evaluation-metrics/02-rag-evaluation.md) -- evaluate your RAG system
