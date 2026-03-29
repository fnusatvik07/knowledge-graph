# Neo4j Fundamentals

**Neo4j** is the most widely adopted graph database in production knowledge graph systems. It stores data as a **property graph** -- nodes and relationships that carry key-value properties -- and provides the **Cypher** query language for expressive pattern matching.

## Why Neo4j for Knowledge Graphs?

- **Native graph storage** -- data is stored as linked structures on disk, so traversals are O(1) per hop rather than requiring expensive joins
- **ACID compliant** -- full transactional guarantees for concurrent reads and writes
- **Cypher query language** -- declarative, SQL-like syntax designed specifically for graph patterns
- **Mature ecosystem** -- drivers for Python, Java, JavaScript, Go; integrations with LangChain, LlamaIndex, and most RAG frameworks
- **Scales to billions** of nodes and relationships in production

---

## The Property Graph Model

Neo4j implements the **labeled property graph** model. Every element in the graph is one of three things:

### Nodes

Nodes represent entities. Each node can have:

- One or more **labels** (like types): `Person`, `Organization`, `Concept`
- Zero or more **properties** (key-value pairs): `{name: "Marie Curie", birth_year: 1867}`

```
(:Person {name: "Marie Curie", birth_year: 1867})
```

### Relationships

Relationships connect two nodes. Each relationship has:

- A **type**: `WORKS_AT`, `DEVELOPED`, `RECEIVED`
- A **direction**: always directed, from one node to another
- Zero or more **properties**: `{since: 1906, role: "Professor"}`

```
(:Person {name: "Marie Curie"})-[:WORKS_AT {since: 1906}]->(:Organization {name: "Sorbonne"})
```

### Properties

Both nodes and relationships can hold properties. Supported types include strings, numbers, booleans, arrays of primitives, temporal types, and spatial points.

### How This Maps to Knowledge Graphs

| Knowledge Graph Concept | Property Graph Equivalent |
|------------------------|--------------------------|
| Entity | Node |
| Entity type | Node label |
| Entity attributes | Node properties |
| Relationship | Relationship |
| Relationship type | Relationship type |
| Relationship attributes | Relationship properties |
| Triple (s, p, o) | `(s)-[:P]->(o)` |

---

## Setting Up Neo4j with Docker

The fastest way to get Neo4j running locally is with Docker.

### Single Container

```bash
docker run \
  --name neo4j-kg \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password_here \
  -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  -d neo4j:5-community
```

### Docker Compose (Recommended)

Create a `docker-compose.yml`:

```yaml
version: "3.8"

services:
  neo4j:
    image: neo4j:5-community
    container_name: neo4j-kg
    ports:
      - "7474:7474"   # Browser UI
      - "7687:7687"   # Bolt protocol
    environment:
      - NEO4J_AUTH=neo4j/knowledge_graph_2024
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
      - NEO4J_server_memory_heap_initial__size=512m
      - NEO4J_server_memory_heap_max__size=2g
      - NEO4J_server_memory_pagecache_size=1g
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.*
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "neo4j", "status"]
      interval: 30s
      timeout: 10s
      retries: 5

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
```

Start the database:

```bash
docker compose up -d
```

Once running, open the **Neo4j Browser** at `http://localhost:7474` and log in with the credentials you set.

---

## Connecting from Python

Install the official driver:

```bash
pip install neo4j
```

### Basic Connection

```python
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "knowledge_graph_2024")

driver = GraphDatabase.driver(URI, auth=AUTH)

# Verify connectivity
driver.verify_connectivity()
print("Connected to Neo4j")
```

### Running Queries

```python
def run_query(driver, query, parameters=None):
    """Execute a Cypher query and return results as a list of dicts."""
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]

# Create a node
run_query(driver, """
    CREATE (p:Person {name: $name, birth_year: $year})
    RETURN p
""", {"name": "Albert Einstein", "year": 1879})

# Query nodes
results = run_query(driver, "MATCH (p:Person) RETURN p.name AS name, p.birth_year AS year")
for row in results:
    print(row)
```

### Transaction Functions (Recommended for Production)

```python
def create_person(tx, name, birth_year):
    tx.run(
        "MERGE (p:Person {name: $name}) SET p.birth_year = $year",
        name=name, year=birth_year
    )

def get_people(tx):
    result = tx.run("MATCH (p:Person) RETURN p.name AS name ORDER BY p.name")
    return [record["name"] for record in result]

with driver.session() as session:
    session.execute_write(create_person, "Niels Bohr", 1885)
    people = session.execute_read(get_people)
    print(people)
```

---

## Indexes and Constraints

Indexes are critical for query performance. Without them, every `MATCH` requires a full scan.

### Create Indexes

```cypher
-- Composite index for common lookups
CREATE INDEX person_name FOR (p:Person) ON (p.name);
CREATE INDEX org_name FOR (o:Organization) ON (o.name);
CREATE INDEX concept_name FOR (c:Concept) ON (c.name);

-- Full-text index for search
CREATE FULLTEXT INDEX entity_search FOR (n:Person|Organization|Concept) ON EACH [n.name, n.description];
```

### Uniqueness Constraints

Constraints enforce data integrity and automatically create indexes.

```cypher
CREATE CONSTRAINT unique_person_name FOR (p:Person) REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT unique_org_name FOR (o:Organization) REQUIRE o.name IS UNIQUE;
```

### List Existing Indexes

```cypher
SHOW INDEXES;
SHOW CONSTRAINTS;
```

---

## ACID Compliance

Neo4j is fully ACID compliant:

| Property | Guarantee |
|----------|-----------|
| **Atomicity** | A transaction either fully completes or fully rolls back |
| **Consistency** | Constraints (uniqueness, existence) are enforced at commit time |
| **Isolation** | Concurrent transactions do not see each other's uncommitted changes |
| **Durability** | Committed data survives crashes (write-ahead log) |

This matters for knowledge graphs in production because:

- Multiple processes can ingest entities and relationships concurrently without corruption
- A failed batch import rolls back cleanly, leaving the graph in a consistent state
- Uniqueness constraints prevent duplicate entities

---

## Loading a Knowledge Graph into Neo4j

### Small Graphs -- Cypher CREATE Statements

```cypher
// Create entities
CREATE (einstein:Person {name: "Albert Einstein", birth_year: 1879})
CREATE (bohr:Person {name: "Niels Bohr", birth_year: 1885})
CREATE (curie:Person {name: "Marie Curie", birth_year: 1867})
CREATE (eth:Organization {name: "ETH Zurich", founded: 1855})
CREATE (relativity:Concept {name: "Theory of Relativity", field: "Physics"})
CREATE (qm:Concept {name: "Quantum Mechanics", field: "Physics"})

// Create relationships
CREATE (einstein)-[:AFFILIATED_WITH {role: "Professor"}]->(eth)
CREATE (einstein)-[:DEVELOPED {year: 1905}]->(relativity)
CREATE (einstein)-[:CONTRIBUTED_TO]->(qm)
CREATE (bohr)-[:DEVELOPED {year: 1913}]->(qm)
CREATE (einstein)-[:DEBATED_WITH {topic: "Quantum Mechanics"}]->(bohr);
```

### Medium Graphs -- CSV Import

Place CSV files in the `import` directory and use `LOAD CSV`:

```cypher
// Load entities from CSV
LOAD CSV WITH HEADERS FROM 'file:///entities.csv' AS row
CALL apoc.create.node([row.label], {name: row.name, description: row.description})
YIELD node
RETURN count(node);

// Load relationships from CSV
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (source {name: row.source})
MATCH (target {name: row.target})
CALL apoc.create.relationship(source, row.rel_type, {}, target)
YIELD rel
RETURN count(rel);
```

### Large Graphs -- neo4j-admin import

For millions of nodes, use the bulk import tool (requires the database to be stopped):

```bash
neo4j-admin database import full \
  --nodes=Person=import/persons_header.csv,import/persons.csv \
  --nodes=Organization=import/orgs_header.csv,import/orgs.csv \
  --relationships=WORKS_AT=import/works_at_header.csv,import/works_at.csv \
  neo4j
```

---

## Neo4j Editions

| Feature | Community (Free) | Enterprise |
|---------|-----------------|------------|
| Core database | Yes | Yes |
| ACID transactions | Yes | Yes |
| Cypher query language | Yes | Yes |
| APOC procedures | Yes | Yes |
| Clustering / HA | No | Yes |
| Role-based access control | No | Yes |
| Online backup | No | Yes |
| Causal clustering | No | Yes |
| License | GPL v3 | Commercial |

For most knowledge graph prototypes and moderate-scale production systems, **Community Edition** is sufficient. Enterprise is needed when you require high availability, fine-grained security, or clustering.

---

## Neo4j AuraDB (Managed Cloud)

If you prefer not to manage infrastructure, **Neo4j AuraDB** is a fully managed cloud service:

- **AuraDB Free** -- limited to 200K nodes, good for learning
- **AuraDB Professional** -- pay-per-use, automatic scaling
- **AuraDB Enterprise** -- dedicated instances, SLA guarantees

```python
# Connecting to AuraDB is the same, just change the URI
URI = "neo4j+s://your-instance.databases.neo4j.io"
AUTH = ("neo4j", "your-aura-password")
driver = GraphDatabase.driver(URI, auth=AUTH)
```

---

## Key Configuration Parameters

| Parameter | Default | Recommendation |
|-----------|---------|---------------|
| `server.memory.heap.initial_size` | 512m | Set to 1/4 of available RAM |
| `server.memory.heap.max_size` | 512m | Set to 1/4 of available RAM |
| `server.memory.pagecache.size` | 512m | Set to remaining RAM after heap and OS |
| `dbms.memory.transaction.total.max` | -- | Limit per-query memory to prevent OOM |

A general rule: give the heap about 25% of RAM, the page cache about 50%, and leave the rest for the OS.

---

## Summary

Neo4j is the production-grade choice for knowledge graphs because it combines:

1. **A natural data model** -- the property graph maps directly to how knowledge graphs are conceptualized
2. **Expressive querying** -- Cypher makes multi-hop traversals readable and maintainable
3. **Transactional safety** -- ACID compliance ensures data integrity under concurrent access
4. **Operational maturity** -- battle-tested in production at companies like NASA, eBay, and Walmart

The next guide covers the **Cypher query language** in depth, with examples tailored to knowledge graph patterns.
