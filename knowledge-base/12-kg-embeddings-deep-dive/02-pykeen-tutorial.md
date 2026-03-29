# PyKEEN Tutorial: Training and Evaluating KG Embeddings

## Overview

PyKEEN (Python KnowlEdge EmbeddiNgs) is the most comprehensive Python library for training and evaluating knowledge graph embedding models. It provides 40+ models, 20+ datasets, and a clean pipeline API for experimentation.

- **Documentation**: https://pykeen.readthedocs.io/
- **Repository**: https://github.com/pykeen/pykeen
- **Paper**: Ali et al. (2021), "PyKEEN 1.0", JMLR

---

## Installation

```bash
# Basic installation
pip install pykeen

# With visualization extras
pip install pykeen[plotting]

# With all extras (MLflow tracking, Optuna HPO, etc.)
pip install pykeen[all]

# Verify installation
python -c "import pykeen; print(pykeen.get_version())"
```

Requirements:
- Python >= 3.8
- PyTorch >= 1.9
- NumPy, Pandas, scikit-learn (installed automatically)

---

## Loading Standard Datasets

PyKEEN ships with many benchmark datasets:

```python
from pykeen.datasets import FB15k237, WN18RR, YAGO310, Nations, Countries

# Load FB15k-237 (Freebase subset)
dataset = FB15k237()

print(f"Training triples: {dataset.training.num_triples}")
print(f"Validation triples: {dataset.validation.num_triples}")
print(f"Testing triples: {dataset.testing.num_triples}")
print(f"Entities: {dataset.num_entities}")
print(f"Relations: {dataset.num_relations}")

# Inspect the triples
print(dataset.training.triples[:5])
# Output: array of [head_label, relation_label, tail_label]
```

### Loading Custom Datasets

```python
from pykeen.triples import TriplesFactory

# From a TSV file (head \t relation \t tail)
tf = TriplesFactory.from_path("my_triples.tsv")

# From a pandas DataFrame
import pandas as pd
df = pd.DataFrame({
    "head": ["Alice", "Bob", "Alice"],
    "relation": ["knows", "works_at", "lives_in"],
    "tail": ["Bob", "Acme", "NYC"]
})
tf = TriplesFactory.from_labeled_triples(
    df[["head", "relation", "tail"]].values
)

# Split into train/val/test
training, testing, validation = tf.split([0.8, 0.1, 0.1])
```

---

## Training a TransE Model

### Using the Pipeline API (Recommended)

The simplest way to run experiments:

```python
from pykeen.pipeline import pipeline

result = pipeline(
    dataset="FB15k-237",
    model="TransE",
    model_kwargs=dict(
        embedding_dim=200,
    ),
    training_kwargs=dict(
        num_epochs=100,
        batch_size=256,
    ),
    optimizer="Adam",
    optimizer_kwargs=dict(
        lr=1e-3,
    ),
    negative_sampler="basic",
    negative_sampler_kwargs=dict(
        num_negatives_per_positive=5,
    ),
    evaluation_kwargs=dict(
        batch_size=128,
    ),
    random_seed=42,
)

# Access results
print(result.metric_results.to_df())
```

### Accessing Metrics

```python
# Get specific metrics
mrr = result.metric_results.get_metric("both.realistic.inverse_harmonic_mean_rank")
hits_at_1 = result.metric_results.get_metric("both.realistic.hits_at_1")
hits_at_3 = result.metric_results.get_metric("both.realistic.hits_at_3")
hits_at_10 = result.metric_results.get_metric("both.realistic.hits_at_10")

print(f"MRR:      {mrr:.4f}")
print(f"Hits@1:   {hits_at_1:.4f}")
print(f"Hits@3:   {hits_at_3:.4f}")
print(f"Hits@10:  {hits_at_10:.4f}")
```

### Saving and Loading Models

```python
# Save the trained pipeline
result.save_to_directory("transE_fb15k237")

# Load later
from pykeen.pipeline import pipeline_from_path
loaded_result = pipeline_from_path("transE_fb15k237")
model = loaded_result.model
```

---

## Training a RotatE Model

```python
result_rotate = pipeline(
    dataset="FB15k-237",
    model="RotatE",
    model_kwargs=dict(
        embedding_dim=200,
    ),
    loss="NSSALoss",  # Self-adversarial negative sampling loss
    loss_kwargs=dict(
        margin=9.0,
        adversarial_temperature=1.0,
    ),
    training_kwargs=dict(
        num_epochs=200,
        batch_size=512,
    ),
    optimizer="Adam",
    optimizer_kwargs=dict(
        lr=1e-4,
    ),
    negative_sampler="basic",
    negative_sampler_kwargs=dict(
        num_negatives_per_positive=64,
    ),
    random_seed=42,
)
```

---

## Training ComplEx and DistMult

```python
# ComplEx
result_complex = pipeline(
    dataset="FB15k-237",
    model="ComplEx",
    model_kwargs=dict(embedding_dim=200),
    loss="BCEAfterSigmoidLoss",
    training_loop="sLCWA",
    training_kwargs=dict(num_epochs=200, batch_size=256),
    optimizer_kwargs=dict(lr=1e-3),
    regularizer="LpRegularizer",
    regularizer_kwargs=dict(p=3, weight=1e-6),
    random_seed=42,
)

# DistMult
result_distmult = pipeline(
    dataset="FB15k-237",
    model="DistMult",
    model_kwargs=dict(embedding_dim=200),
    loss="BCEAfterSigmoidLoss",
    training_loop="sLCWA",
    training_kwargs=dict(num_epochs=200, batch_size=256),
    optimizer_kwargs=dict(lr=1e-3),
    regularizer="LpRegularizer",
    regularizer_kwargs=dict(p=3, weight=1e-5),
    random_seed=42,
)
```

---

## Evaluation Metrics Explained

### Mean Reciprocal Rank (MRR)

```
MRR = (1/|Q|) * sum_{i=1}^{|Q|} (1 / rank_i)
```

- Ranges from 0 to 1 (higher is better)
- Heavily penalizes low ranks (rank 1 contributes 1.0, rank 10 contributes 0.1)
- The primary metric for KG embedding evaluation

### Hits@K

```
Hits@K = (1/|Q|) * sum_{i=1}^{|Q|} I(rank_i <= K)
```

- Fraction of queries where the correct entity appears in the top K
- Hits@1 ~ exact accuracy, Hits@10 ~ recall at 10

### Adjusted Mean Rank (AMR)

```
AMR = MR / expected_MR_random
```

- Normalized version of Mean Rank, accounts for dataset size
- Ranges from 0 to 2 (lower is better, 1 = random)

---

## Link Prediction

Once trained, use the model for predictions:

```python
from pykeen.predict import predict_target

# Predict tail: (Barack Obama, born_in, ?)
predictions = predict_target(
    model=result.model,
    head="Barack Obama",
    relation="born_in",
    triples_factory=result.training,
)
# Returns DataFrame with entity, score, in_training columns
print(predictions.df.head(10))

# Predict head: (?, born_in, Hawaii)
predictions = predict_target(
    model=result.model,
    tail="Hawaii",
    relation="born_in",
    triples_factory=result.training,
)
print(predictions.df.head(10))

# Predict relation: (Barack Obama, ?, Michelle Obama)
predictions = predict_target(
    model=result.model,
    head="Barack Obama",
    tail="Michelle Obama",
    triples_factory=result.training,
)
print(predictions.df.head(10))
```

### Scoring Specific Triples

```python
import torch

# Get entity/relation IDs
entity_to_id = result.training.entity_to_id
relation_to_id = result.training.relation_to_id

h_id = entity_to_id["Barack Obama"]
r_id = relation_to_id["born_in"]
t_id = entity_to_id["Hawaii"]

h_tensor = torch.tensor([[h_id]])
r_tensor = torch.tensor([[r_id]])
t_tensor = torch.tensor([[t_id]])

score = result.model.score_hrt(
    torch.cat([h_tensor, r_tensor, t_tensor], dim=1)
)
print(f"Score: {score.item():.4f}")
```

---

## Visualizing Embeddings

### Extract Embeddings

```python
import numpy as np

# Get entity embeddings as numpy array
entity_embeddings = result.model.entity_representations[0]()
entity_emb_np = entity_embeddings.detach().cpu().numpy()

# For complex models (RotatE, ComplEx), embeddings have real and imaginary parts
# Use the real part or concatenate both
if entity_emb_np.dtype == np.complex64:
    entity_emb_np = np.concatenate([entity_emb_np.real, entity_emb_np.imag], axis=1)
```

### t-SNE Visualization

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(entity_emb_np[:500])  # subset for speed

plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
plt.title("Entity Embeddings (t-SNE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.savefig("entity_embeddings_tsne.png", dpi=150)
plt.show()
```

### UMAP Visualization

```python
import umap

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
embeddings_2d = reducer.fit_transform(entity_emb_np[:500])

plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
plt.title("Entity Embeddings (UMAP)")
plt.tight_layout()
plt.savefig("entity_embeddings_umap.png", dpi=150)
plt.show()
```

---

## Comparing Models on the Same Dataset

```python
from pykeen.pipeline import pipeline
import pandas as pd

models = ["TransE", "RotatE", "ComplEx", "DistMult"]
results_list = []

for model_name in models:
    print(f"\nTraining {model_name}...")
    result = pipeline(
        dataset="FB15k-237",
        model=model_name,
        model_kwargs=dict(embedding_dim=200),
        training_kwargs=dict(num_epochs=100, batch_size=256),
        optimizer_kwargs=dict(lr=1e-3),
        random_seed=42,
    )

    mrr = result.metric_results.get_metric("both.realistic.inverse_harmonic_mean_rank")
    h1 = result.metric_results.get_metric("both.realistic.hits_at_1")
    h3 = result.metric_results.get_metric("both.realistic.hits_at_3")
    h10 = result.metric_results.get_metric("both.realistic.hits_at_10")

    results_list.append({
        "Model": model_name,
        "MRR": mrr,
        "Hits@1": h1,
        "Hits@3": h3,
        "Hits@10": h10,
    })

comparison_df = pd.DataFrame(results_list)
print("\n=== Model Comparison ===")
print(comparison_df.to_string(index=False))
```

### Expected Results (approximate, FB15k-237)

| Model    | MRR   | Hits@1 | Hits@3 | Hits@10 |
|----------|-------|--------|--------|---------|
| TransE   | 0.29  | 0.20   | 0.32   | 0.47    |
| RotatE   | 0.33  | 0.24   | 0.37   | 0.53    |
| ComplEx  | 0.32  | 0.23   | 0.35   | 0.51    |
| DistMult | 0.28  | 0.19   | 0.30   | 0.45    |

---

## Hyperparameter Optimization

PyKEEN integrates with Optuna for HPO:

```python
from pykeen.hpo import hpo_pipeline

hpo_result = hpo_pipeline(
    dataset="FB15k-237",
    model="TransE",
    n_trials=30,
    training_kwargs=dict(num_epochs=100),
    model_kwargs_ranges=dict(
        embedding_dim=dict(type=int, low=100, high=500, q=100),
    ),
    optimizer_kwargs_ranges=dict(
        lr=dict(type=float, low=1e-4, high=1e-2, log=True),
    ),
    negative_sampler_kwargs_ranges=dict(
        num_negatives_per_positive=dict(type=int, low=1, high=100, log=True),
    ),
)

print(f"Best trial MRR: {hpo_result.objective:.4f}")
print(f"Best params: {hpo_result.best_trial.params}")
```

---

## Tips and Best Practices

1. **Start small**: Use `Nations` or `Countries` datasets for quick debugging before scaling to FB15k-237
2. **Monitor training**: Use `result.losses` to plot loss curves and detect convergence
3. **Embedding dimension**: 200 is a good default; increase for larger KGs
4. **Negative sampling**: More negatives generally helps (at the cost of training time)
5. **Learning rate**: Start with 1e-3 for Adam; reduce to 1e-4 for RotatE
6. **Early stopping**: Use validation MRR with patience of 5-10 epochs:

```python
from pykeen.stoppers import EarlyStopper

result = pipeline(
    dataset="FB15k-237",
    model="TransE",
    stopper="early",
    stopper_kwargs=dict(
        metric="inverse_harmonic_mean_rank",
        patience=10,
        frequency=5,
    ),
    training_kwargs=dict(num_epochs=500),
)
```

---

## References

- PyKEEN documentation: https://pykeen.readthedocs.io/
- PyKEEN benchmarks: https://github.com/pykeen/pykeen#benchmarks
- Ali, M., et al. (2021). "PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings." JMLR.
