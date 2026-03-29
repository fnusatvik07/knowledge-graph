# Cypher Query Language for Knowledge Graphs

**Cypher** is Neo4j's declarative query language, designed to make graph pattern matching intuitive and readable. If you can draw a relationship on a whiteboard, you can express it in Cypher.

## Core Syntax Pattern

Cypher uses ASCII art to represent graph patterns:

```
(node)-[:RELATIONSHIP]->(other_node)
```

- Parentheses `()` represent **nodes**
- Square brackets `[]` represent **relationships**
- Arrows `-->` or `<--` indicate **direction**

---

## CREATE -- Building the Graph

### Creating Nodes

```cypher
// Create a single node with a label and properties
CREATE (p:Person {name: "Alan Turing", birth_year: 1912, field: "Computer Science"})

// Create multiple nodes in one statement
CREATE (org:Organization {name: "Bletchley Park", country: "United Kingdom"})
CREATE (concept:Concept {name: "Turing Machine", year_introduced: 1936})
CREATE (concept2:Concept {name: "Enigma Decryption", classification: "Cryptography"})
```

### Creating Nodes with Multiple Labels

A node can have more than one label. This is useful when an entity fits multiple categories.

```cypher
CREATE (p:Person:Scientist {name: "Ada Lovelace", birth_year: 1815})
```

### Creating Relationships

```cypher
// First create the nodes, then connect them
MATCH (turing:Person {name: "Alan Turing"})
MATCH (bp:Organization {name: "Bletchley Park"})
CREATE (turing)-[:WORKED_AT {from: 1938, to: 1945, role: "Codebreaker"}]->(bp)
```

### Creating Nodes and Relationships Together

```cypher
CREATE (turing:Person {name: "Alan Turing"})
  -[:DEVELOPED {year: 1936}]->
  (tm:Concept {name: "Turing Machine"})
```

---

## MATCH -- Reading the Graph

`MATCH` is the primary read clause. It describes a pattern and Neo4j finds all subgraphs that fit.

### Match All Nodes of a Type

```cypher
MATCH (p:Person)
RETURN p.name, p.birth_year
ORDER BY p.birth_year
```

### Match a Specific Node

```cypher
MATCH (p:Person {name: "Alan Turing"})
RETURN p
```

### Match Relationships

```cypher
// Find all people and the organizations they work at
MATCH (p:Person)-[:WORKED_AT]->(org:Organization)
RETURN p.name AS person, org.name AS organization
```

### Match Any Relationship Type

```cypher
// Find everything connected to Alan Turing
MATCH (p:Person {name: "Alan Turing"})-[r]->(target)
RETURN type(r) AS relationship, target.name AS connected_to
```

---

## WHERE -- Filtering Results

### Property Comparisons

```cypher
// People born before 1900
MATCH (p:Person)
WHERE p.birth_year < 1900
RETURN p.name, p.birth_year
```

### String Matching

```cypher
// Names containing "Einstein"
MATCH (p:Person)
WHERE p.name CONTAINS "Einstein"
RETURN p.name

// Names starting with "A"
MATCH (p:Person)
WHERE p.name STARTS WITH "A"
RETURN p.name

// Regex matching
MATCH (p:Person)
WHERE p.name =~ ".*Curie.*"
RETURN p.name
```

### Combining Conditions

```cypher
MATCH (p:Person)-[:AFFILIATED_WITH]->(org:Organization)
WHERE p.birth_year > 1850 AND org.country = "Germany"
RETURN p.name, org.name
```

### Existence Checks

```cypher
// Find people who have NOT received any award
MATCH (p:Person)
WHERE NOT EXISTS {
    MATCH (p)-[:RECEIVED]->(:Award)
}
RETURN p.name
```

### List Predicates

```cypher
// Find people whose field is in a given list
MATCH (p:Person)
WHERE p.field IN ["Physics", "Chemistry", "Mathematics"]
RETURN p.name, p.field
```

---

## RETURN -- Shaping Output

### Aliasing

```cypher
MATCH (p:Person)-[r:DEVELOPED]->(c:Concept)
RETURN p.name AS scientist,
       c.name AS contribution,
       r.year AS year_developed
```

### Distinct Results

```cypher
MATCH (p:Person)-[:AFFILIATED_WITH]->(org:Organization)
RETURN DISTINCT org.name AS organization
```

### Ordering and Limiting

```cypher
MATCH (p:Person)
RETURN p.name, p.birth_year
ORDER BY p.birth_year DESC
LIMIT 10
```

### Collecting into Lists

```cypher
MATCH (p:Person)-[:DEVELOPED]->(c:Concept)
RETURN p.name AS scientist, collect(c.name) AS contributions
```

---

## MERGE -- Idempotent Create

`MERGE` is essential for knowledge graph construction. It creates a node or relationship only if it does not already exist, preventing duplicates.

### Merge Nodes

```cypher
// Create the node if it doesn't exist, match it if it does
MERGE (p:Person {name: "Albert Einstein"})
ON CREATE SET p.birth_year = 1879, p.created_at = datetime()
ON MATCH SET p.last_seen = datetime()
RETURN p
```

### Merge Relationships

```cypher
MATCH (p:Person {name: "Albert Einstein"})
MATCH (org:Organization {name: "ETH Zurich"})
MERGE (p)-[r:AFFILIATED_WITH]->(org)
ON CREATE SET r.role = "Professor", r.since = 1912
RETURN r
```

### Why MERGE Matters for Knowledge Graphs

When building a KG from extracted triples, the same entity or relationship may be extracted multiple times from different source documents. `MERGE` ensures you get one clean graph instead of scattered duplicates.

```cypher
// Processing extracted triples from an NLP pipeline
UNWIND $triples AS triple
MERGE (s {name: triple.subject})
SET s:Entity
MERGE (o {name: triple.object})
SET o:Entity
MERGE (s)-[r:RELATES_TO {type: triple.predicate}]->(o)
RETURN count(r) AS relationships_created
```

---

## DELETE and REMOVE -- Modifying the Graph

### Delete a Relationship

```cypher
MATCH (p:Person {name: "Alan Turing"})-[r:WORKED_AT]->(org:Organization {name: "Bletchley Park"})
DELETE r
```

### Delete a Node (Must Delete Relationships First)

```cypher
// This will fail if the node has relationships
MATCH (p:Person {name: "Alan Turing"})
DELETE p

// Use DETACH DELETE to remove the node and all its relationships
MATCH (p:Person {name: "Alan Turing"})
DETACH DELETE p
```

### Remove a Property

```cypher
MATCH (p:Person {name: "Albert Einstein"})
REMOVE p.temporary_field
```

### Remove a Label

```cypher
MATCH (p:Person:Scientist {name: "Ada Lovelace"})
REMOVE p:Scientist
```

---

## Pattern Matching -- The Power of Cypher

### Variable-Length Paths

Find entities connected within a range of hops.

```cypher
// Find all concepts reachable from Einstein within 1 to 3 hops
MATCH path = (p:Person {name: "Albert Einstein"})-[*1..3]->(target)
RETURN target.name, length(path) AS hops
```

### Named Paths

```cypher
// Capture the entire path as a variable
MATCH path = (p:Person {name: "Marie Curie"})-[:DEVELOPED]->(:Concept)-[:RELATED_TO]->(:Concept)
RETURN nodes(path) AS entities, relationships(path) AS connections
```

### Multi-Hop Knowledge Graph Queries

These queries demonstrate the real power of graph databases for knowledge graphs.

**"Who collaborated with people that contributed to Quantum Mechanics?"**

```cypher
MATCH (collaborator:Person)-[:COLLABORATED_WITH]->(contributor:Person)
      -[:CONTRIBUTED_TO]->(c:Concept {name: "Quantum Mechanics"})
RETURN collaborator.name AS collaborator,
       contributor.name AS contributor
```

**"What organizations employ researchers who published papers on topics related to Machine Learning?"**

```cypher
MATCH (org:Organization)<-[:WORKS_AT]-(researcher:Person)
      -[:PUBLISHED]->(paper:Publication)-[:COVERS]->(topic:Concept)
      -[:RELATED_TO]->(ml:Concept {name: "Machine Learning"})
RETURN org.name AS organization,
       researcher.name AS researcher,
       paper.title AS paper,
       topic.name AS topic
```

**"Find the chain of influence from Aristotle to modern AI"**

```cypher
MATCH path = (ancient:Person {name: "Aristotle"})
      -[:INFLUENCED*1..6]->(modern:Person)
      -[:CONTRIBUTED_TO]->(ai:Concept {name: "Artificial Intelligence"})
RETURN [n IN nodes(path) | n.name] AS influence_chain,
       length(path) AS degrees_of_separation
ORDER BY length(path)
LIMIT 5
```

### Shortest Path

```cypher
// Find the shortest connection between two entities
MATCH path = shortestPath(
    (a:Person {name: "Marie Curie"})-[*]-(b:Person {name: "Alan Turing"})
)
RETURN [n IN nodes(path) | n.name] AS path_nodes,
       [r IN relationships(path) | type(r)] AS relationship_types,
       length(path) AS distance
```

### All Shortest Paths

```cypher
MATCH paths = allShortestPaths(
    (a:Person {name: "Albert Einstein"})-[*]-(b:Concept {name: "Artificial Intelligence"})
)
RETURN [n IN nodes(paths) | n.name] AS path_nodes,
       length(paths) AS distance
```

---

## Aggregations

### Counting

```cypher
// Count entities by type
MATCH (n)
RETURN labels(n)[0] AS entity_type, count(n) AS count
ORDER BY count DESC
```

### Grouping

```cypher
// Contributions per scientist
MATCH (p:Person)-[:DEVELOPED|CONTRIBUTED_TO]->(c:Concept)
RETURN p.name AS scientist,
       count(c) AS num_contributions,
       collect(c.name) AS contributions
ORDER BY num_contributions DESC
```

### Statistical Aggregations

```cypher
MATCH (p:Person)
RETURN min(p.birth_year) AS earliest_born,
       max(p.birth_year) AS latest_born,
       avg(p.birth_year) AS average_birth_year,
       count(p) AS total_people
```

### Relationship Statistics

```cypher
// Most connected entities (highest degree)
MATCH (n)
WITH n, size([(n)-[]-() | 1]) AS degree
RETURN n.name, labels(n)[0] AS type, degree
ORDER BY degree DESC
LIMIT 10
```

---

## WITH -- Chaining Query Parts

`WITH` acts as a pipe, passing results from one part of the query to the next.

```cypher
// Find prolific scientists and then find their organizations
MATCH (p:Person)-[:DEVELOPED]->(c:Concept)
WITH p, count(c) AS contributions
WHERE contributions > 2
MATCH (p)-[:AFFILIATED_WITH]->(org:Organization)
RETURN p.name, contributions, collect(org.name) AS organizations
ORDER BY contributions DESC
```

---

## UNWIND -- Working with Lists

`UNWIND` expands a list into rows, useful for batch operations.

```cypher
// Batch create entities from a list
WITH ["Physics", "Chemistry", "Biology", "Mathematics"] AS fields
UNWIND fields AS field
MERGE (f:Field {name: field})
RETURN f.name
```

### Batch Import Triples

```cypher
// Import knowledge graph triples in bulk
UNWIND [
    {subject: "Albert Einstein", predicate: "DEVELOPED", object: "Theory of Relativity"},
    {subject: "Albert Einstein", predicate: "RECEIVED", object: "Nobel Prize in Physics"},
    {subject: "Niels Bohr", predicate: "DEVELOPED", object: "Bohr Model"},
    {subject: "Marie Curie", predicate: "DISCOVERED", object: "Radium"}
] AS triple
MERGE (s:Entity {name: triple.subject})
MERGE (o:Entity {name: triple.object})
WITH s, o, triple
CALL apoc.create.relationship(s, triple.predicate, {}, o) YIELD rel
RETURN count(rel)
```

---

## Practical Patterns for Knowledge Graphs

### Retrieve Context for an LLM

Given an entity, retrieve its local neighborhood as structured context.

```cypher
// Get 2-hop neighborhood around an entity
MATCH (center {name: $entity_name})-[r1]-(neighbor1)
OPTIONAL MATCH (neighbor1)-[r2]-(neighbor2)
WHERE neighbor2 <> center
RETURN center.name AS entity,
       type(r1) AS rel1, neighbor1.name AS hop1,
       type(r2) AS rel2, neighbor2.name AS hop2
LIMIT 100
```

### Find Communities

```cypher
// Identify densely connected clusters
MATCH (p1:Person)-[:COLLABORATED_WITH]-(p2:Person)
WITH p1, collect(DISTINCT p2.name) AS collaborators, count(DISTINCT p2) AS collab_count
WHERE collab_count > 3
RETURN p1.name, collaborators, collab_count
ORDER BY collab_count DESC
```

### Temporal Queries

```cypher
// Timeline of a person's career
MATCH (p:Person {name: "Albert Einstein"})-[r]->(target)
WHERE r.year IS NOT NULL
RETURN type(r) AS event, target.name AS entity, r.year AS year
ORDER BY r.year
```

---

## Performance Tips

1. **Always create indexes** on properties used in `MATCH` and `WHERE` clauses
2. **Use parameters** (`$name`) instead of string concatenation to enable query plan caching
3. **Use `MERGE` instead of `CREATE`** to prevent duplicate entities
4. **Limit variable-length paths** -- `[*1..3]` is fine, `[*]` without bounds can be catastrophic
5. **Profile your queries** with `PROFILE` or `EXPLAIN` to see the execution plan

```cypher
// See the query execution plan
PROFILE
MATCH (p:Person)-[:DEVELOPED]->(c:Concept)
WHERE p.birth_year > 1800
RETURN p.name, c.name
```

---

## Summary

| Clause | Purpose | Example |
|--------|---------|---------|
| `CREATE` | Add new nodes/relationships | `CREATE (p:Person {name: "X"})` |
| `MATCH` | Find existing patterns | `MATCH (p:Person)-[:KNOWS]->(q)` |
| `WHERE` | Filter matched results | `WHERE p.age > 30` |
| `RETURN` | Specify output columns | `RETURN p.name AS name` |
| `MERGE` | Create if not exists | `MERGE (p:Person {name: "X"})` |
| `DELETE` | Remove nodes/relationships | `DETACH DELETE p` |
| `SET` | Update properties | `SET p.age = 31` |
| `WITH` | Chain query stages | `WITH p, count(*) AS c` |
| `UNWIND` | Expand lists into rows | `UNWIND $list AS item` |
| `ORDER BY` | Sort results | `ORDER BY p.name DESC` |

Cypher's pattern-matching syntax makes it the most readable way to query knowledge graphs. Combine it with Neo4j's native graph storage, and you have a powerful platform for both building and querying knowledge graphs at scale.
