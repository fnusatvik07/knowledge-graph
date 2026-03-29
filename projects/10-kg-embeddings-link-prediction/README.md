# Project 10: KG Embeddings & Link Prediction

Train knowledge graph embedding models and predict missing links. This project fills the biggest gap in the repo by covering the mathematical foundations of KG representation learning.

## What This Project Does

1. **Load Benchmark Data** -- Load FB15k-237, the standard KG embedding benchmark derived from Freebase. Print dataset statistics and splits.
2. **Train TransE** -- Train a TransE model (translational distance) with PyKEEN. Evaluate with MRR, Hits@1, Hits@3, Hits@10.
3. **Train RotatE** -- Train a RotatE model (rotational embeddings in complex space). Compare with TransE.
4. **Train ComplEx** -- Train a ComplEx model (complex-valued bilinear). Compare all three architectures.
5. **Model Comparison** -- Side-by-side comparison of all models with bar charts of MRR and Hits@K metrics.
6. **Link Prediction** -- Use the best model to predict missing links: top-K tail entities for (h, r, ?) and top-K head entities for (?, r, t).
7. **Custom KG Embeddings** -- Train embeddings on a custom small KG (AI/tech domain), bridging the gap between benchmarks and real-world use.
8. **Visualize Embeddings** -- Reduce entity embeddings to 2D with t-SNE, color by entity type, show clustering behavior.

## Key Concepts

### KG Embedding Models
- **TransE**: Models relations as translations in embedding space: h + r ~ t
- **RotatE**: Models relations as rotations in complex space: t ~ h * r
- **ComplEx**: Uses complex-valued embeddings with Hermitian dot product scoring

### Evaluation Metrics
- **MRR** (Mean Reciprocal Rank): Average of 1/rank for correct predictions
- **Hits@K**: Fraction of correct entities ranked in top K (K = 1, 3, 10)
- Evaluated under the **filtered** setting (removing other known true triples)

## Prerequisites

- Python 3.11+
- Dependencies: `pykeen`, `torch`, `matplotlib`, `scikit-learn`, `numpy`, `pandas`
- Install PyKEEN: `pip install pykeen`
- GPU recommended for training on FB15k-237 but CPU works for smaller datasets

## Quick Start

```bash
# Load and explore FB15k-237
python src/01_load_benchmark.py

# Train embedding models (each takes 5-30 min depending on hardware)
python src/02_train_transe.py
python src/03_train_rotate.py
python src/04_train_complex.py

# Compare all models
python src/05_model_comparison.py

# Predict missing links
python src/06_link_prediction.py

# Train on custom KG
python src/07_custom_kg_embeddings.py

# Visualize embeddings
python src/08_visualize_embeddings.py
```

## File Structure

```
10-kg-embeddings-link-prediction/
├── README.md
├── data/
│   └── custom_kg.tsv
├── output/
└── src/
    ├── __init__.py
    ├── 01_load_benchmark.py
    ├── 02_train_transe.py
    ├── 03_train_rotate.py
    ├── 04_train_complex.py
    ├── 05_model_comparison.py
    ├── 06_link_prediction.py
    ├── 07_custom_kg_embeddings.py
    └── 08_visualize_embeddings.py
```
