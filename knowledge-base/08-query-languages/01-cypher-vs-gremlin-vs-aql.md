# Cypher vs Gremlin vs AQL vs GSQL vs SPARQL

Choosing a graph query language is often tied to your choice of graph database. This section compares the five major graph query languages side by side, showing the same queries expressed in each to help you understand their trade-offs.

## Language Overview

### Cypher (Neo4j, Memgraph, FalkorDB)

Cypher is a declarative, pattern-matching query language. You describe the shape of the graph pattern you want to find, and the engine figures out how to find it.

```cypher
// Basic pattern: find a person who works at an organization
MATCH (p:Person)-[:WORKS_AT]->(o:Organization)
RETURN p.name, o.name
```

- **Style**: Declarative, SQL-like
- **Strength**: Intuitive ASCII-art pattern syntax
- **Databases**: Neo4j, Memgraph, FalkorDB (OpenCypher), Amazon Neptune (partial)
- **Standard**: openCypher (open specification), GQL (ISO standard, Cypher-based)

### Gremlin (Apache TinkerPop)

Gremlin is an imperative, traversal-based language. You describe the steps of a traversal through the graph.

```groovy
// Basic traversal: find a person who works at an organization
g.V().hasLabel('Person').as('p')
  .out('WORKS_AT').hasLabel('Organization').as('o')
  .select('p', 'o').by('name').by('name')
```

- **Style**: Imperative, step-by-step traversal
- **Strength**: Fine-grained control over traversal, wide database support
- **Databases**: Amazon Neptune, JanusGraph, Azure Cosmos DB, TinkerPop-compatible databases
- **Standard**: Apache TinkerPop framework

### AQL (ArangoDB)

AQL is a multi-model query language that handles documents, graphs, and key-value lookups in a single syntax.

```aql
// Basic query: find a person who works at an organization
FOR p IN persons
    FOR o IN organizations
        FILTER LENGTH(
            FOR e IN works_at
                FILTER e._from == p._id AND e._to == o._id
                RETURN e
        ) > 0
        RETURN { person: p.name, org: o.name }
```

Or using graph traversal syntax:

```aql
FOR p IN persons
    FOR v, e IN 1..1 OUTBOUND p works_at
        RETURN { person: p.name, org: v.name }
```

- **Style**: Declarative, SQL-like with graph extensions
- **Strength**: Multi-model (graph + document + key-value in one query)
- **Databases**: ArangoDB
- **Standard**: Proprietary to ArangoDB

### GSQL (TigerGraph)

GSQL is a SQL-like language designed for massively parallel graph analytics.

```gsql
// Basic query: find a person who works at an organization
CREATE QUERY find_employees() FOR GRAPH MyGraph {
    Start = {Person.*};
    Result = SELECT o
        FROM Start:p -(WORKS_AT:e)- Organization:o
        RETURN p.name, o.name;
    PRINT Result;
}
```

- **Style**: SQL-like with procedural blocks
- **Strength**: Massive parallelism, built-in accumulator pattern for analytics
- **Databases**: TigerGraph
- **Standard**: Proprietary to TigerGraph

### SPARQL (RDF Databases)

SPARQL queries RDF (Resource Description Framework) triple stores. It uses subject-predicate-object patterns.

```sparql
# Basic query: find a person who works at an organization
PREFIX ex: <http://example.org/>
SELECT ?personName ?orgName
WHERE {
    ?person a ex:Person ;
            ex:name ?personName ;
            ex:worksAt ?org .
    ?org a ex:Organization ;
         ex:name ?orgName .
}
```

- **Style**: Declarative, triple-pattern matching
- **Strength**: Semantic web, linked data, federated queries across endpoints
- **Databases**: Apache Jena, Blazegraph, Virtuoso, GraphDB, Wikidata
- **Standard**: W3C standard

## Side-by-Side Query Comparison

**Scenario**: "Find all people who work at organizations located in California"

### Cypher

```cypher
MATCH (p:Person)-[:WORKS_AT]->(o:Organization)-[:LOCATED_IN]->(l:Location {name: 'California'})
RETURN p.name AS person, o.name AS organization
ORDER BY p.name
```

### Gremlin

```groovy
g.V().has('Location', 'name', 'California')
  .in('LOCATED_IN').hasLabel('Organization').as('o')
  .in('WORKS_AT').hasLabel('Person').as('p')
  .select('p', 'o')
  .by('name').by('name')
  .order().by(select('p'))
```

### AQL

```aql
FOR l IN locations
    FILTER l.name == "California"
    FOR o IN 1..1 INBOUND l located_in
        FOR p IN 1..1 INBOUND o works_at
            SORT p.name
            RETURN { person: p.name, organization: o.name }
```

### GSQL

```gsql
CREATE QUERY people_in_california() FOR GRAPH MyGraph {
    california = SELECT l FROM Location:l WHERE l.name == "California";
    orgs = SELECT o FROM california:l -(reverse_LOCATED_IN:e)- Organization:o;
    people = SELECT p FROM orgs:o -(reverse_WORKS_AT:e)- Person:p
        ORDER BY p.name;
    PRINT people;
}
```

### SPARQL

```sparql
PREFIX ex: <http://example.org/>
SELECT ?personName ?orgName
WHERE {
    ?person a ex:Person ;
            ex:name ?personName ;
            ex:worksAt ?org .
    ?org a ex:Organization ;
         ex:name ?orgName ;
         ex:locatedIn ?location .
    ?location ex:name "California" .
}
ORDER BY ?personName
```

## Advanced Query: Shortest Path

**Scenario**: "Find the shortest path between Alice and Bob"

### Cypher

```cypher
MATCH path = shortestPath(
    (a:Person {name: 'Alice'})-[*]-(b:Person {name: 'Bob'})
)
RETURN path, length(path) AS hops
```

### Gremlin

```groovy
g.V().has('Person', 'name', 'Alice')
  .repeat(both().simplePath())
  .until(has('Person', 'name', 'Bob'))
  .path()
  .limit(1)
```

### AQL

```aql
FOR v, e IN OUTBOUND SHORTEST_PATH
    'persons/alice' TO 'persons/bob'
    GRAPH 'myGraph'
    RETURN { vertex: v, edge: e }
```

### GSQL

```gsql
CREATE QUERY shortest_path_ab() FOR GRAPH MyGraph {
    Start = {Person.*};
    Start = SELECT s FROM Start:s WHERE s.name == "Alice";
    Result = SELECT t FROM Start:s -(:e)- :t
        WHERE t.name == "Bob"
        ACCUM @@path += e;
    PRINT Result;
}
```

### SPARQL

```sparql
# SPARQL does not have built-in shortest path (not a graph traversal language).
# Property paths provide limited traversal:
PREFIX ex: <http://example.org/>
SELECT ?person
WHERE {
    ex:Alice (ex:knows)+ ?person .
    FILTER (?person = ex:Bob)
}
```

## Comparison Table

| Feature | Cypher | Gremlin | AQL | GSQL | SPARQL |
|---------|--------|---------|-----|------|--------|
| **Paradigm** | Declarative | Imperative | Declarative | Declarative + Procedural | Declarative |
| **Learning Curve** | Low | Medium | Medium | Medium-High | Medium |
| **Pattern Matching** | Excellent | Manual | Good | Good | Triple patterns |
| **Multi-hop Traversals** | Good | Excellent | Good | Excellent | Limited |
| **Aggregation** | SQL-like | Step-based | SQL-like | Accumulator-based | GROUP BY |
| **Shortest Path** | Built-in | Manual (repeat/until) | Built-in | Manual | Not supported |
| **Write Operations** | CREATE/MERGE | addV/addE | INSERT/UPDATE | INSERT | INSERT DATA |
| **Multi-model** | Graph only | Graph only | Graph + Document + KV | Graph only | RDF triples |
| **Parallelism** | Single-machine | Framework-dependent | Moderate | Massive | Endpoint-dependent |
| **Open Standard** | openCypher/GQL | Apache TinkerPop | Proprietary | Proprietary | W3C |

## Using LLMs to Generate Graph Queries

All of these languages can be generated by LLMs from natural language. LangChain provides integrations for several:

```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Natural language -> Cypher -> results -> natural language answer
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-4o", temperature=0),
    graph=graph,
)
result = chain.invoke({"query": "Who works at organizations in California?"})
```

> **LangChain Graph integrations**: https://python.langchain.com/docs/integrations/graphs/
> **GraphCypherQAChain**: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

## Choosing a Query Language

| If you need... | Choose... | Because... |
|---------------|-----------|-----------|
| Easiest to learn | Cypher | ASCII-art patterns are intuitive |
| Widest cloud support | Gremlin | Supported by AWS, Azure, many vendors |
| Multi-model queries | AQL | Graph + document + KV in one language |
| Large-scale analytics | GSQL | Designed for parallel processing |
| Semantic web / linked data | SPARQL | W3C standard for RDF |
| LLM generation support | Cypher | Most LLM training data, best LangChain integration |

## Next Steps

- [02 - Natural Language to Graph Queries](./02-natural-language-to-graph-queries.md) -- using LLMs to write queries for you
- [Cypher Query Language](../03-graph-storage/03-cypher-query-language.md) -- deep dive into Cypher
