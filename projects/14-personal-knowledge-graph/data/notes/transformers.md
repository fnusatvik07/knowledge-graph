# Transformers

The transformer architecture, introduced in the 2017 paper "Attention Is All You Need," has become the dominant architecture in [[deep_learning]]. It replaced recurrent networks with a purely attention-based mechanism, enabling massive parallelization and scaling.

## The Attention Mechanism

The core innovation is self-attention, which allows each token in a sequence to attend to every other token. This captures long-range dependencies that RNNs struggle with.

Key components:
- **Query, Key, Value**: Each input is projected into three vectors. Attention weights are computed as softmax(QK^T / sqrt(d_k))V
- **Multi-head attention**: Multiple attention heads capture different types of relationships
- **Positional encoding**: Since transformers have no inherent notion of order, position information is added via sinusoidal or learned embeddings

## Architecture Details

A transformer encoder-decoder consists of:
- **Encoder**: Stack of self-attention + feed-forward layers. Each layer has residual connections and layer normalization.
- **Decoder**: Similar to encoder but with masked self-attention (preventing future token access) and cross-attention to encoder outputs.

## Landmark Models

- **BERT**: Encoder-only, pretrained with masked language modeling. Excels at understanding tasks.
- **GPT series**: Decoder-only, autoregressive. Powers ChatGPT and modern language models.
- **T5**: Encoder-decoder, treats every NLP task as text-to-text.
- **Vision Transformers (ViT)**: Apply transformers to image patches, rivaling CNNs.

## Impact on NLP

Transformers have unified [[deep_learning]] approaches to NLP. Pre-training on large corpora followed by task-specific fine-tuning has become the standard paradigm. This has enabled breakthrough performance on translation, summarization, question answering, and code generation.

#transformers #attention #nlp #deep-learning
