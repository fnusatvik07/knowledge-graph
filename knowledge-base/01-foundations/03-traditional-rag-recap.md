# Traditional RAG Recap

Before understanding *why* we need Graph RAG, let's recap how traditional (vector-based) RAG works and where it falls short.

## How Traditional RAG Works

The standard RAG pipeline has three stages:

### 1. Indexing (Offline)
```
Documents → Chunk → Embed → Store in Vector DB
```
- Split documents into text chunks (typically 500-1000 characters)
- Generate an embedding vector for each chunk using a model like `text-embedding-3-small`
- Store the vectors in a vector database (ChromaDB, Pinecone, Weaviate, etc.)

### 2. Retrieval (Query Time)
```
Query → Embed → Find Top-K Similar Chunks
```
- Convert the user's question into an embedding
- Perform approximate nearest neighbor search in the vector DB
- Return the top-K most similar chunks

### 3. Generation (Query Time)
```
[System prompt] + [Retrieved chunks] + [User question] → LLM → Answer
```
- Concatenate retrieved chunks as context
- Send to the LLM with the user's question
- Generate a grounded answer

## What Traditional RAG Does Well

- **Direct fact lookup**: "What is the capital of France?" — finds the chunk containing "Paris is the capital of France"
- **Semantic matching**: Understands that "heart attack" and "myocardial infarction" are related, even without exact keyword match
- **Simplicity**: Easy to implement, well-supported by frameworks, fast retrieval
- **Scalability**: Vector databases handle millions of chunks efficiently

## Where Traditional RAG Fails

### 1. Cross-Document Reasoning
**Question**: "What themes connect the research of all scientists mentioned in these papers?"

Traditional RAG retrieves individual chunks but has no way to synthesize information *across* multiple documents. It returns fragments, not connections.

### 2. Multi-Hop Questions
**Question**: "Which companies were founded by alumni of the university where the inventor of the telephone studied?"

This requires: inventor of telephone → Alexander Graham Bell → University of Edinburgh → alumni → companies they founded. No single chunk contains this chain.

### 3. Global Summarization
**Question**: "What are the main topics discussed across this entire document corpus?"

Top-K retrieval returns a few similar chunks, not a holistic summary. The answer is inherently distributed across the entire corpus.

### 4. Relationship-Centric Questions
**Question**: "How are protein X and disease Y connected?"

The connection might span multiple papers, each contributing one link in the chain. Vector similarity can't discover multi-step relationships.

### 5. Contradictory Information
When different sources provide conflicting information, traditional RAG may retrieve both without any mechanism to reason about which is more recent, authoritative, or contextually appropriate.

## The Gap That Graph RAG Fills

| Capability | Vector RAG | Graph RAG |
|-----------|-----------|-----------|
| Direct fact lookup | Great | Good |
| Semantic matching | Great | Good (with embeddings) |
| Cross-document reasoning | Poor | Great |
| Multi-hop questions | Poor | Great |
| Global summarization | Poor | Great |
| Relationship discovery | Poor | Great |
| Speed | Fast | Slower (more processing) |
| Cost | Low | Higher (LLM extraction) |

Graph RAG doesn't *replace* vector RAG — it *complements* it. The most effective production systems in 2026 use **hybrid retrieval**: vector search for direct fact lookup + graph traversal for relationship reasoning.

## Key Takeaways

- Traditional RAG: chunk → embed → retrieve similar → generate
- Excels at direct fact lookup and semantic matching
- Fails at multi-hop reasoning, cross-document synthesis, and global summarization
- Graph RAG addresses these gaps by adding a structured knowledge layer
- Best approach: hybrid (vector + graph) retrieval
