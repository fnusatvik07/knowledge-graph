# Natural Language to Graph Queries

One of the most powerful applications of LLMs in the knowledge graph space is converting natural language questions into executable graph queries. This eliminates the need for users to learn Cypher, Gremlin, or AQL, and makes graph data accessible to everyone.

> **LangChain Graph Integrations**: https://python.langchain.com/docs/integrations/graphs/

## The Text2Cypher Pattern

Text2Cypher is the most mature natural-language-to-graph-query approach, primarily because Cypher is well-represented in LLM training data and has intuitive pattern-matching syntax.

### Basic Architecture

```
User Question
    |
    v
Schema Injection (node labels, relationship types, properties)
    |
    v
LLM (generates Cypher query)
    |
    v
Validation & Safety Check
    |
    v
Execute against Graph Database
    |
    v
LLM (formats results as natural language answer)
    |
    v
Response to User
```

### LangChain GraphCypherQAChain

LangChain provides a ready-made chain for this exact pattern:

```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Connect to Neo4j and auto-detect schema
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# Print the auto-detected schema (sent to the LLM)
print(graph.schema)
# Node properties: Person {name: STRING, age: INTEGER}, Organization {name: STRING} ...
# Relationships: WORKS_AT, LOCATED_IN, FOUNDED ...

# Create the QA chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,  # shows the generated Cypher
    return_intermediate_steps=True,
)

# Ask a question
result = chain.invoke({"query": "Who founded organizations in San Francisco?"})
print(result["result"])
# "Acme Corp was founded by John Smith and TechStart was founded by Jane Doe"
print(result["intermediate_steps"])
# Shows the generated Cypher query
```

> **GraphCypherQAChain docs**: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

### Custom Text2Cypher with LangChain

For more control, build your own chain:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o", temperature=0)

cypher_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Cypher query expert. Given a graph schema and a natural language question,
generate a valid Cypher query.

Graph Schema:
{schema}

Rules:
- Only use node labels, relationship types, and properties from the schema
- Always use parameters for literal values when possible
- Use LIMIT to prevent returning too many results
- Only generate READ queries (no CREATE, DELETE, SET, MERGE, DROP)
- Return only the Cypher query, no explanation"""),
    ("human", "{question}")
])

cypher_chain = cypher_prompt | llm | StrOutputParser()

schema = """
Node labels: Person (name, title, department), Organization (name, industry, founded_year), Location (name, state, country)
Relationships: WORKS_AT (Person->Organization, since), LOCATED_IN (Organization->Location), REPORTS_TO (Person->Person)
"""

cypher_query = cypher_chain.invoke({
    "schema": schema,
    "question": "Find all engineers who work at companies in the technology industry"
})
print(cypher_query)
# MATCH (p:Person)-[:WORKS_AT]->(o:Organization)-[:LOCATED_IN]->(l:Location)
# WHERE p.title CONTAINS 'Engineer' AND o.industry = 'Technology'
# RETURN p.name, o.name, l.name
# LIMIT 25
```

> **LangChain prompt templates**: https://python.langchain.com/docs/how_to/prompts_composition/

## Prompt Engineering for Query Generation

### Schema Injection

Always include the full graph schema in the prompt. Without it, the LLM will guess at node labels and properties.

```python
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# Auto-generated schema string
schema = graph.schema
# Includes: node labels, properties with types, relationship types, property constraints
```

### Few-Shot Examples

Include example question-query pairs for your specific domain:

```python
few_shot_examples = """
Example 1:
Question: "How many employees does each department have?"
Cypher: MATCH (p:Person)-[:WORKS_IN]->(d:Department) RETURN d.name, count(p) AS employee_count ORDER BY employee_count DESC

Example 2:
Question: "Who reports to Sarah Chen?"
Cypher: MATCH (p:Person)-[:REPORTS_TO]->(m:Person {{name: 'Sarah Chen'}}) RETURN p.name, p.title

Example 3:
Question: "What is the longest chain of reporting relationships?"
Cypher: MATCH path = (p:Person)-[:REPORTS_TO*]->(top:Person) WHERE NOT (top)-[:REPORTS_TO]->() RETURN length(path) AS chain_length, [n IN nodes(path) | n.name] AS chain ORDER BY chain_length DESC LIMIT 5
"""
```

### Handling Ambiguity

Natural language is inherently ambiguous. Build disambiguation into your prompt:

```python
disambiguation_prompt = """When the question is ambiguous:
1. If a name could match multiple entities, use case-insensitive CONTAINS instead of exact match
2. If the relationship direction is unclear, try both directions
3. If the question could mean multiple things, generate the most common interpretation
4. If you cannot generate a valid query, respond with: CANNOT_GENERATE: <reason>

Example ambiguous query:
Question: "Show me connections to Apple"
Note: "Apple" could be a company, a product, or a person's name.
Cypher: MATCH (n)-[r]-(m) WHERE toLower(m.name) CONTAINS 'apple' RETURN n, type(r), m LIMIT 25
"""
```

## Natural Language to Other Query Languages

### Text2Gremlin

```python
gremlin_prompt = ChatPromptTemplate.from_messages([
    ("system", """Generate a Gremlin traversal query for Apache TinkerPop.

Schema:
{schema}

Rules:
- Use g.V() to start vertex traversals
- Use .has(label, property, value) for filtering
- Use .out(), .in(), .both() for traversals
- Use .valueMap() or .values() for returning properties
- Always include .limit() to prevent unbounded results"""),
    ("human", "{question}")
])
```

### Text2AQL

```python
aql_prompt = ChatPromptTemplate.from_messages([
    ("system", """Generate an AQL query for ArangoDB.

Schema:
Collections: {collections}
Edge Collections: {edge_collections}
Graph: {graph_name}

Rules:
- Use FOR ... IN collection for document iteration
- Use FOR v, e IN 1..N OUTBOUND/INBOUND start_vertex edge_collection for traversals
- Use FILTER for conditions
- Use LIMIT to restrict results
- Use RETURN to specify output shape"""),
    ("human", "{question}")
])
```

> **LangChain ArangoDB integration**: https://python.langchain.com/docs/integrations/graphs/arangodb/

## Validation and Safety

Generated queries must be validated before execution to prevent data loss, injection attacks, and runaway queries.

### Read-Only Enforcement

```python
import re

DANGEROUS_PATTERNS = [
    r'\bCREATE\b', r'\bDELETE\b', r'\bDETACH\b', r'\bDROP\b',
    r'\bSET\b', r'\bREMOVE\b', r'\bMERGE\b', r'\bCALL\b.*\bdbms\b',
]

def validate_cypher_readonly(query: str) -> bool:
    """Ensure a Cypher query is read-only."""
    upper_query = query.upper()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, upper_query):
            return False
    return True

def safe_execute(graph, query: str):
    if not validate_cypher_readonly(query):
        raise ValueError(f"Query contains write operations and was blocked: {query}")
    return graph.query(query)
```

### Result Size Limits

```python
def add_limit_if_missing(query: str, max_rows: int = 100) -> str:
    """Add a LIMIT clause if the query doesn't have one."""
    if 'LIMIT' not in query.upper():
        query = query.rstrip().rstrip(';')
        query += f"\nLIMIT {max_rows}"
    return query
```

### Timeout Protection

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def execute_with_timeout(query: str, timeout_ms: int = 5000):
    """Execute a query with a timeout to prevent runaway queries."""
    with driver.session() as session:
        result = session.run(query, timeout=timeout_ms)
        return [record.data() for record in result]
```

## Evaluation: How Good Are LLM-Generated Queries?

Evaluate your text2cypher pipeline by building a test set:

```python
test_cases = [
    {
        "question": "How many people work at Google?",
        "expected_cypher": "MATCH (p:Person)-[:WORKS_AT]->(o:Organization {name: 'Google'}) RETURN count(p)",
        "expected_answer_contains": ["42"]  # or whatever the real count is
    },
    {
        "question": "Who are the top 5 most connected people?",
        "expected_cypher": "MATCH (p:Person)-[r]-() RETURN p.name, count(r) AS connections ORDER BY connections DESC LIMIT 5",
        "expected_answer_contains": ["connections"]
    }
]

def evaluate_text2cypher(chain, test_cases, graph):
    results = []
    for tc in test_cases:
        try:
            response = chain.invoke({"query": tc["question"]})
            generated_cypher = response.get("intermediate_steps", [{}])[0].get("query", "")
            answer = response["result"]

            # Check if the answer contains expected content
            answer_correct = all(
                expected in answer for expected in tc["expected_answer_contains"]
            )

            results.append({
                "question": tc["question"],
                "generated_cypher": generated_cypher,
                "answer": answer,
                "answer_correct": answer_correct,
            })
        except Exception as e:
            results.append({
                "question": tc["question"],
                "error": str(e),
                "answer_correct": False,
            })

    accuracy = sum(1 for r in results if r.get("answer_correct")) / len(results)
    print(f"Accuracy: {accuracy:.1%}")
    return results
```

> **LangChain evaluation**: https://python.langchain.com/docs/how_to/#evaluation

## Best Practices

1. **Always inject the schema** -- without it, LLMs hallucinate node labels and properties
2. **Include few-shot examples** from your specific domain
3. **Enforce read-only** for user-facing queries unless write access is explicitly needed
4. **Add LIMIT clauses** to prevent returning millions of rows
5. **Set query timeouts** to prevent runaway traversals
6. **Log generated queries** for debugging and evaluation
7. **Use GPT-4o or Claude for complex queries** -- smaller models struggle with multi-hop patterns
8. **Validate before executing** -- never run unvalidated LLM output against your database

## Next Steps

- [01 - Cypher vs Gremlin vs AQL](./01-cypher-vs-gremlin-vs-aql.md) -- understand the query languages themselves
- [Cypher Query Language](../03-graph-storage/03-cypher-query-language.md) -- deep dive into Cypher
- [LLM Extraction Patterns](../02-kg-construction/04-llm-extraction-patterns.md) -- structured output for extraction
