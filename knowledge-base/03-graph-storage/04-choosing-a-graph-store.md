# Choosing a Graph Store

Selecting the right storage backend for your knowledge graph depends on your use case, scale, and operational requirements. This guide covers the most common options and when to use each.

---

## Decision Framework

Ask these four questions to narrow down your choice:

1. **How large is your graph?** (thousands, millions, or billions of nodes)
2. **Do you need concurrent access?** (single user vs. multi-user / multi-service)
3. **What are your persistence requirements?** (ephemeral analysis vs. durable production store)
4. **What is your budget for infrastructure?** (zero, moderate, cloud-scale)

---

## Option 1: NetworkX (In-Memory Python)

### What It Is

A pure Python library that represents graphs as dictionaries of dictionaries. No server, no setup, no infrastructure.

### When to Use It

- **Prototyping and experimentation** -- validate your ontology, test extraction pipelines, iterate quickly
- **Small graphs** -- up to approximately 100K nodes and 500K edges fit comfortably in memory
- **One-off analysis** -- run centrality, community detection, or pathfinding algorithms once
- **Educational projects** -- learn graph concepts without operational overhead
- **Embedding in applications** -- when the graph is part of a larger Python process and does not need to be shared

### When NOT to Use It

- Your graph exceeds available RAM
- Multiple processes or services need to read/write the graph simultaneously
- You need durable storage that survives process restarts without manual serialization
- You need a query language (NetworkX uses Python API calls, not a declarative language)

### Practical Limits

| Metric | Comfortable | Possible but Slow | Not Feasible |
|--------|-------------|-------------------|--------------|
| Nodes | < 100K | 100K - 1M | > 1M |
| Edges | < 500K | 500K - 5M | > 5M |
| RAM usage | < 2 GB | 2 - 16 GB | > 16 GB |
| Concurrent users | 1 | 1 | N/A |

### Cost

Free and open source. No infrastructure required.

---

## Option 2: Neo4j (Property Graph Database)

### What It Is

A dedicated graph database with native graph storage, ACID transactions, and the Cypher query language. The most widely adopted graph database in production.

### When to Use It

- **Production knowledge graphs** -- serving queries to applications and APIs
- **Medium to large graphs** -- millions to low billions of nodes
- **Concurrent access** -- multiple services or users querying and updating simultaneously
- **Complex traversals** -- multi-hop queries, shortest path, pattern matching across the graph
- **Integration with RAG pipelines** -- LangChain, LlamaIndex, and most frameworks have Neo4j connectors
- **When you need a query language** -- Cypher is far more expressive than Python API calls for graph queries

### When NOT to Use It

- Quick throwaway experiments where NetworkX is faster to set up
- Extremely simple graphs that a relational database could handle
- Graphs exceeding tens of billions of edges where a distributed solution is needed

### Editions

| Feature | Community (Free) | AuraDB Free | AuraDB Pro | Enterprise |
|---------|-----------------|-------------|------------|------------|
| Max nodes | Unlimited | 200K | Unlimited | Unlimited |
| ACID | Yes | Yes | Yes | Yes |
| Clustering | No | N/A | Yes | Yes |
| RBAC | No | No | Yes | Yes |
| Cost | $0 | $0 | Pay-per-use | License fee |

### Cost

- **Community Edition**: Free, self-hosted
- **AuraDB Free**: Free, cloud-hosted, limited to 200K nodes
- **AuraDB Professional**: Starts around $65/month
- **Enterprise**: Contact sales

---

## Option 3: Amazon Neptune

### What It Is

A fully managed graph database service on AWS. Supports both the property graph model (via openCypher and Gremlin) and RDF (via SPARQL).

### When to Use It

- **AWS-native architecture** -- your infrastructure is already on AWS and you want tight integration with IAM, VPC, CloudWatch, and S3
- **RDF / SPARQL requirement** -- you are building a semantic web application or need W3C-standard RDF support
- **Managed operations** -- you do not want to manage database servers, backups, or failover
- **High availability** -- Neptune replicates across availability zones automatically
- **Large graphs** -- designed for billions of edges

### When NOT to Use It

- You are not on AWS (vendor lock-in is significant)
- Budget is limited -- Neptune has no free tier beyond a short trial
- You need the full Cypher feature set (Neptune supports openCypher, which is a subset)
- You want a local development environment (Neptune only runs in AWS)

### Cost

Starts at approximately $0.348/hour for the smallest instance (db.r5.large). There is no free tier for ongoing use.

---

## Option 4: Memgraph

### What It Is

An in-memory graph database compatible with Cypher. Designed for real-time, low-latency graph queries.

### When to Use It

- **Real-time queries** -- sub-millisecond latency requirements (fraud detection, recommendation engines)
- **Streaming data** -- Memgraph integrates natively with Kafka and Pulsar for graph updates from event streams
- **Cypher compatibility** -- if you know Cypher from Neo4j, you can use it directly in Memgraph
- **Cost-sensitive production** -- Memgraph's community edition is open source with fewer restrictions than Neo4j Community

### When NOT to Use It

- Your graph does not fit in RAM (Memgraph is memory-first)
- You need the mature ecosystem and tooling of Neo4j (APOC, GDS library, Browser UI)
- You need enterprise support guarantees (Memgraph's ecosystem is smaller)

### Cost

- **Community**: Free, open source (no feature restrictions)
- **Enterprise**: Commercial license, contact for pricing

---

## Option 5: Other Alternatives

### Kuzu (Embedded Graph Database)

An embedded (in-process) graph database with Cypher support. Think "SQLite for graphs."

- Best for: single-user analytical workloads, embedded in a Python application
- Cypher-compatible query language
- Columnar storage for fast analytical queries
- No server to manage

### ArangoDB (Multi-Model)

Supports documents, graphs, and key-value storage in a single database.

- Best for: teams that need graph queries alongside document storage
- Query language: AQL (not Cypher)
- Good for hybrid use cases where some data is better modeled as documents

### TigerGraph (Distributed Graph Analytics)

Purpose-built for large-scale graph analytics.

- Best for: graphs with hundreds of billions of edges, heavy analytics workloads
- Query language: GSQL (proprietary)
- Steeper learning curve but massive scale

### RDF / Triple Stores (Apache Jena, Blazegraph, Stardog)

For W3C Semantic Web standards (RDF, OWL, SPARQL).

- Best for: formal ontologies, linked data, regulatory compliance
- Query language: SPARQL
- Stronger focus on reasoning and inference rules than property graph databases

---

## Comparison Table

| Criteria | NetworkX | Neo4j | Amazon Neptune | Memgraph | Kuzu |
|----------|----------|-------|---------------|----------|------|
| **Type** | Library | Database | Managed service | Database | Embedded DB |
| **Storage** | In-memory | Disk + cache | Managed | In-memory | Disk (columnar) |
| **Query interface** | Python API | Cypher | openCypher / SPARQL | Cypher | Cypher |
| **Max practical scale** | ~100K nodes | Billions | Billions | Fits in RAM | Billions |
| **ACID transactions** | No | Yes | Yes | Yes | Yes |
| **Concurrent access** | No | Yes | Yes | Yes | No (embedded) |
| **Setup complexity** | `pip install` | Docker / Cloud | AWS Console | Docker | `pip install` |
| **Persistence** | Manual (file) | Automatic | Automatic | Snapshot-based | Automatic |
| **Cost** | Free | Free - $$$ | $$$ | Free - $$ | Free |
| **Python integration** | Native | Driver | Driver (boto3) | Driver | Native |
| **RAG framework support** | Manual | Excellent | Good | Growing | Limited |
| **Best for** | Prototyping | Production KGs | AWS-native prod | Real-time | Analytical |

---

## Recommended Path

For most knowledge graph projects, the following progression works well:

### Phase 1: Prototype with NetworkX

```
pip install networkx
```

- Build your entity extraction and relationship extraction pipeline
- Store results in a NetworkX DiGraph
- Validate your ontology -- are the entity types and relationship types right?
- Run graph algorithms to verify the graph structure makes sense
- Serialize to JSON or GraphML for inspection

### Phase 2: Move to Neo4j for Production

```
docker compose up -d  # Neo4j Community Edition
```

- Import your validated graph into Neo4j
- Write Cypher queries for your application's access patterns
- Add indexes on frequently queried properties
- Connect your RAG pipeline (LangChain's `Neo4jGraph`, LlamaIndex's `KnowledgeGraphIndex`)
- Set up monitoring and backups

### Phase 3: Scale if Needed

Only reach for Neptune, TigerGraph, or a distributed solution when:

- Your graph exceeds what a single Neo4j instance can handle
- You need multi-region replication
- You need sub-millisecond latency that requires an in-memory solution like Memgraph

---

## Decision Flowchart

```
Is this a prototype or experiment?
├── YES --> NetworkX
└── NO --> Do you need concurrent access or persistence?
           ├── NO --> NetworkX or Kuzu
           └── YES --> Are you on AWS and want managed infrastructure?
                       ├── YES --> Amazon Neptune
                       └── NO --> Do you need real-time / sub-ms latency?
                                  ├── YES --> Memgraph
                                  └── NO --> Neo4j
```

---

## Migration Considerations

### NetworkX to Neo4j

Export your NetworkX graph as Cypher statements or CSV files.

```python
# Generate Cypher from NetworkX (see 01-networkx-quickstart.md for full function)
for node, attrs in graph.nodes(data=True):
    label = attrs.get("type", "Entity")
    print(f'MERGE (n:{label} {{name: "{node}"}})')

for source, target, attrs in graph.edges(data=True):
    rel = attrs.get("relation", "RELATED_TO").upper()
    print(f'MATCH (a {{name: "{source}"}}), (b {{name: "{target}"}}) MERGE (a)-[:{rel}]->(b)')
```

### Neo4j to Neptune

Neptune supports openCypher, so most read queries translate directly. Use `neptune-export` to dump Neo4j and `neptune-bulk-loader` to import.

### Key Migration Pitfalls

1. **Property types** -- Neo4j supports temporal and spatial types that not all databases handle
2. **APOC dependency** -- if you use APOC procedures heavily, verify equivalents exist in your target
3. **Index syntax** -- each database has its own index creation syntax
4. **Transaction semantics** -- moving from ACID (Neo4j) to eventually consistent systems requires application changes

---

## Summary

There is no single "best" graph store. The right choice depends on where you are in your project lifecycle:

- **Just starting out?** Use **NetworkX**. Zero friction, rich algorithms, perfect for learning.
- **Building a real application?** Use **Neo4j**. Battle-tested, excellent Cypher language, strong ecosystem.
- **Running on AWS at scale?** Consider **Amazon Neptune** for managed operations.
- **Need real-time performance?** Look at **Memgraph** for in-memory speed.
- **Want embedded, no server?** Try **Kuzu** as the SQLite of graph databases.

Start simple, validate your data model, and scale your infrastructure only when the data demands it.
