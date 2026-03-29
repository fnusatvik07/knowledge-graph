# Knowledge Graph + Cypher — Complete Live Demo

Open Neo4j Browser: **http://localhost:7474**
Login: `neo4j` / `workshop2024`

Copy-paste each code block into the Neo4j query bar and press **Play (▶)**.

---

## Step 1: Clear Everything

Start with a clean database.

```cypher
MATCH (n) DETACH DELETE n
```

---

## Step 2: Create People

A **node** = a circle in the graph. Each node has a **label** (type) and **properties** (details).

```cypher
CREATE (sam:Person {name: "Sam Altman", role: "CEO", age: 39})
CREATE (dario:Person {name: "Dario Amodei", role: "CEO", age: 41})
CREATE (jensen:Person {name: "Jensen Huang", role: "CEO", age: 61})
CREATE (demis:Person {name: "Demis Hassabis", role: "CEO", age: 48})
CREATE (harrison:Person {name: "Harrison Chase", role: "CEO", age: 30})
CREATE (yann:Person {name: "Yann LeCun", role: "Chief AI Scientist", age: 63})
CREATE (ilya:Person {name: "Ilya Sutskever", role: "Co-Founder", age: 37})
RETURN sam, dario, jensen, demis, harrison, yann, ilya
```

> You should see **7 circles**. Click any node to see its properties.

---

## Step 3: Create Companies

Same syntax, different label. Now we have two types of nodes in the same graph.

```cypher
CREATE (openai:Company {name: "OpenAI", founded: 2015, hq: "San Francisco", valuation_B: 150})
CREATE (anthropic:Company {name: "Anthropic", founded: 2021, hq: "San Francisco", valuation_B: 60})
CREATE (nvidia:Company {name: "NVIDIA", founded: 1993, hq: "Santa Clara", valuation_B: 3000})
CREATE (deepmind:Company {name: "Google DeepMind", founded: 2010, hq: "London", valuation_B: 0})
CREATE (langchain:Company {name: "LangChain", founded: 2022, hq: "San Francisco", valuation_B: 0.2})
CREATE (meta:Company {name: "Meta AI", founded: 2013, hq: "Menlo Park", valuation_B: 0})
CREATE (ssi:Company {name: "Safe Superintelligence", founded: 2024, hq: "Palo Alto", valuation_B: 5})
RETURN openai, anthropic, nvidia, deepmind, langchain, meta, ssi
```

> Now you have **7 people + 7 companies = 14 nodes**. But they're all disconnected.

---

## Step 4: Create Products

```cypher
CREATE (gpt4:Product {name: "GPT-4", type: "LLM", released: 2023})
CREATE (claude:Product {name: "Claude", type: "LLM", released: 2023})
CREATE (cuda:Product {name: "CUDA", type: "Platform", released: 2007})
CREATE (alphafold:Product {name: "AlphaFold", type: "Protein Prediction", released: 2020})
CREATE (langgraph:Product {name: "LangGraph", type: "Agent Framework", released: 2024})
CREATE (llama:Product {name: "Llama", type: "Open Source LLM", released: 2023})
RETURN gpt4, claude, cuda, alphafold, langgraph, llama
```

> **20 nodes total** — but all disconnected circles. A knowledge graph needs **relationships** (lines between circles).

---

## Step 5: Create Relationships

This is where the graph comes alive. We **connect** the nodes with **typed, directed relationships**.

### Who leads which company

```cypher
MATCH (sam:Person {name: "Sam Altman"}), (openai:Company {name: "OpenAI"})
CREATE (sam)-[:LEADS]->(openai)
```

```cypher
MATCH (dario:Person {name: "Dario Amodei"}), (anthropic:Company {name: "Anthropic"})
CREATE (dario)-[:LEADS]->(anthropic)
```

```cypher
MATCH (jensen:Person {name: "Jensen Huang"}), (nvidia:Company {name: "NVIDIA"})
CREATE (jensen)-[:LEADS]->(nvidia)
```

```cypher
MATCH (demis:Person {name: "Demis Hassabis"}), (deepmind:Company {name: "Google DeepMind"})
CREATE (demis)-[:LEADS]->(deepmind)
```

```cypher
MATCH (harrison:Person {name: "Harrison Chase"}), (langchain:Company {name: "LangChain"})
CREATE (harrison)-[:LEADS]->(langchain)
```

```cypher
MATCH (yann:Person {name: "Yann LeCun"}), (meta:Company {name: "Meta AI"})
CREATE (yann)-[:WORKS_AT]->(meta)
```

```cypher
MATCH (ilya:Person {name: "Ilya Sutskever"}), (ssi:Company {name: "Safe Superintelligence"})
CREATE (ilya)-[:FOUNDED]->(ssi)
```

### Career history (who previously worked where)

Relationships can have **properties** too — like role and years.

```cypher
MATCH (dario:Person {name: "Dario Amodei"}), (openai:Company {name: "OpenAI"})
CREATE (dario)-[:PREVIOUSLY_AT {role: "VP of Research", years: "2016-2021"}]->(openai)
```

```cypher
MATCH (ilya:Person {name: "Ilya Sutskever"}), (openai:Company {name: "OpenAI"})
CREATE (ilya)-[:PREVIOUSLY_AT {role: "Chief Scientist", years: "2015-2024"}]->(openai)
```

### Which company built which product

```cypher
MATCH (openai:Company {name: "OpenAI"}), (gpt4:Product {name: "GPT-4"})
CREATE (openai)-[:BUILT]->(gpt4)
```

```cypher
MATCH (anthropic:Company {name: "Anthropic"}), (claude:Product {name: "Claude"})
CREATE (anthropic)-[:BUILT]->(claude)
```

```cypher
MATCH (nvidia:Company {name: "NVIDIA"}), (cuda:Product {name: "CUDA"})
CREATE (nvidia)-[:BUILT]->(cuda)
```

```cypher
MATCH (deepmind:Company {name: "Google DeepMind"}), (alphafold:Product {name: "AlphaFold"})
CREATE (deepmind)-[:BUILT]->(alphafold)
```

```cypher
MATCH (langchain:Company {name: "LangChain"}), (langgraph:Product {name: "LangGraph"})
CREATE (langchain)-[:BUILT]->(langgraph)
```

```cypher
MATCH (meta:Company {name: "Meta AI"}), (llama:Product {name: "Llama"})
CREATE (meta)-[:BUILT]->(llama)
```

### Cross-company dependencies (who uses whose product)

```cypher
MATCH (openai:Company {name: "OpenAI"}), (cuda:Product {name: "CUDA"})
CREATE (openai)-[:USES]->(cuda)
```

```cypher
MATCH (anthropic:Company {name: "Anthropic"}), (cuda:Product {name: "CUDA"})
CREATE (anthropic)-[:USES]->(cuda)
```

```cypher
MATCH (deepmind:Company {name: "Google DeepMind"}), (cuda:Product {name: "CUDA"})
CREATE (deepmind)-[:USES]->(cuda)
```

```cypher
MATCH (meta:Company {name: "Meta AI"}), (cuda:Product {name: "CUDA"})
CREATE (meta)-[:USES]->(cuda)
```

### Investment relationships

```cypher
MATCH (nvidia:Company {name: "NVIDIA"}), (openai:Company {name: "OpenAI"})
CREATE (nvidia)-[:INVESTED_IN]->(openai)
```

```cypher
MATCH (nvidia:Company {name: "NVIDIA"}), (anthropic:Company {name: "Anthropic"})
CREATE (nvidia)-[:INVESTED_IN]->(anthropic)
```

### Now see the full graph

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m
```

> This is your **knowledge graph**. Notice how CUDA connects to 4 companies — that's NVIDIA's moat. Notice Dario has two connections to OpenAI (he used to work there AND his company uses NVIDIA's CUDA which NVIDIA invested in OpenAI).

---

## Step 6: Simple Queries

### Who leads which company?

```cypher
MATCH (p:Person)-[:LEADS]->(c:Company)
RETURN p.name AS person, c.name AS company
```

### Which product does everyone depend on?

```cypher
MATCH (c:Company)-[:USES]->(p:Product)
RETURN p.name AS product, collect(c.name) AS used_by, count(c) AS num_users
ORDER BY num_users DESC
```

### Where did Dario work before Anthropic?

```cypher
MATCH (d:Person {name: "Dario Amodei"})-[:PREVIOUSLY_AT]->(c:Company)
RETURN c.name AS previous_company
```

### Companies founded after 2020

```cypher
MATCH (c:Company)
WHERE c.founded > 2020
RETURN c.name, c.founded
ORDER BY c.founded
```

---

## Step 7: Multi-Hop Queries

**This is why knowledge graphs exist.** These questions require traversing multiple relationships. SQL can't do this. Vector search can't do this.

### What connects Dario Amodei to NVIDIA? (3 hops)

Path: Dario → LEADS → Anthropic → USES → CUDA ← BUILT ← NVIDIA

```cypher
MATCH path = (dario:Person {name: "Dario Amodei"})-[*1..3]-(nvidia:Company {name: "NVIDIA"})
RETURN path
```

> You'll see the full chain of connections visualized.

### Shortest path: Harrison Chase to Jensen Huang

```cypher
MATCH path = shortestPath(
  (a:Person {name: "Harrison Chase"})-[*]-(b:Person {name: "Jensen Huang"})
)
RETURN [n IN nodes(path) | n.name] AS route, length(path) AS hops
```

> Try this: can you guess the path before running it?

### Everyone within 2 hops of OpenAI

```cypher
MATCH (openai:Company {name: "OpenAI"})-[*1..2]-(connected)
WHERE connected:Person
RETURN DISTINCT connected.name AS person
```

### Companies sharing the same dependency

```cypher
MATCH (c1:Company)-[:USES]->(p:Product)<-[:USES]-(c2:Company)
WHERE c1.name < c2.name
RETURN c1.name AS company1, c2.name AS company2, p.name AS shared_dependency
```

---

## Step 8: Update and Delete

### Add a property to an existing node

```cypher
MATCH (openai:Company {name: "OpenAI"})
SET openai.employees = 3500
RETURN openai
```

### Add a new node and connect it

```cypher
CREATE (elon:Person {name: "Elon Musk", role: "Former Board Member"})
WITH elon
MATCH (openai:Company {name: "OpenAI"})
CREATE (elon)-[:CO_FOUNDED]->(openai)
RETURN elon
```

### Delete a relationship

```cypher
MATCH (elon:Person {name: "Elon Musk"})-[r:CO_FOUNDED]->(openai:Company {name: "OpenAI"})
DELETE r
```

### Delete a node and all its relationships

```cypher
MATCH (elon:Person {name: "Elon Musk"})
DETACH DELETE elon
```

### Delete everything (reset the database)

```cypher
MATCH (n) DETACH DELETE n
```

---

## Step 9: Aggregation

### Count nodes by type

```cypher
MATCH (n)
RETURN labels(n)[0] AS type, count(n) AS total
ORDER BY total DESC
```

### Count relationships by type

```cypher
MATCH ()-[r]->()
RETURN type(r) AS relationship, count(r) AS total
ORDER BY total DESC
```

### Most connected entities

```cypher
MATCH (n)
OPTIONAL MATCH (n)-[r]-()
RETURN n.name AS entity, labels(n)[0] AS type, count(r) AS connections
ORDER BY connections DESC
LIMIT 10
```

---

## Recap

What we built:

| Component | Count |
|-----------|-------|
| People | 7 |
| Companies | 7 |
| Products | 6 |
| Relationships | 20+ |

Types of queries we ran:

| Query Type | Example | Can SQL do it? | Can vector search do it? |
|-----------|---------|---------------|------------------------|
| Simple lookup | Who leads OpenAI? | Yes | Yes |
| Filter | Companies after 2020 | Yes | No |
| Aggregation | Most used product | Yes | No |
| Multi-hop | Dario → NVIDIA path | **No** | **No** |
| Shortest path | Harrison → Jensen | **No** | **No** |
| Shared dependencies | Companies using same product | Complex joins | **No** |

**Key insight**: The last three rows are why knowledge graphs exist. Only a graph can traverse relationships across multiple entities to discover connections.

**Next**: We'll let an LLM build these graphs **automatically** from unstructured text documents — no manual CREATE statements needed.
