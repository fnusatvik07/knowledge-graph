# Cypher — The Graph Query Language

Cypher is to **Neo4j** what SQL is to **PostgreSQL**. It's how you talk to a graph database.

---

## The 5 Core Commands

| Command | What it does | SQL equivalent |
|---------|-------------|---------------|
| `CREATE` | Make a node or relationship | `INSERT INTO` |
| `MATCH` | Find nodes or patterns | `SELECT FROM` |
| `RETURN` | Show results | `SELECT` columns |
| `SET` | Update properties | `UPDATE SET` |
| `DELETE` | Remove nodes or relationships | `DELETE FROM` |

---

## Syntax Basics

### Nodes = Round Brackets `()`

```
(p)                          — anonymous node
(p:Person)                   — node with label "Person"
(p:Person {name: "Alice"})   — node with label + property
```

### Relationships = Square Brackets `[]` with Arrow `-->`

```
(a)-[r]->(b)                 — relationship from a to b
(a)-[:KNOWS]->(b)            — relationship with type
(a)-[:KNOWS {since: 2020}]->(b)  — relationship with property
```

### The Pattern

```
(node)-[:RELATIONSHIP]->(node)
```

That's it. Everything in Cypher is about matching or creating **this pattern**.

---

## CREATE — Make Things

### Create a node

```cypher
CREATE (a:Person {name: "Alice", age: 30})
RETURN a
```

> A circle appears. That's Alice.

### Create another node

```cypher
CREATE (b:Person {name: "Bob", age: 25})
RETURN b
```

### Create a relationship between them

```cypher
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
CREATE (a)-[:FRIENDS_WITH]->(b)
RETURN a, b
```

> A line connects Alice to Bob. The arrow shows direction.

### Create everything at once

```cypher
CREATE (c:Person {name: "Charlie", age: 35})-[:WORKS_WITH]->(a:Person {name: "Alice"})
RETURN c, a
```

---

## MATCH + RETURN — Find Things

### Find all nodes

```cypher
MATCH (n)
RETURN n
```

### Find by label

```cypher
MATCH (p:Person)
RETURN p.name, p.age
```

### Find by property

```cypher
MATCH (p:Person {name: "Alice"})
RETURN p
```

### Find with WHERE

```cypher
MATCH (p:Person)
WHERE p.age > 28
RETURN p.name, p.age
```

### Find relationships

```cypher
MATCH (a:Person)-[r:FRIENDS_WITH]->(b:Person)
RETURN a.name, b.name
```

### Find any relationship

```cypher
MATCH (a)-[r]->(b)
RETURN a.name, type(r) AS relationship, b.name
```

---

## SET — Update Things

### Add or change a property

```cypher
MATCH (a:Person {name: "Alice"})
SET a.city = "New York"
RETURN a
```

### Add multiple properties

```cypher
MATCH (b:Person {name: "Bob"})
SET b.city = "London", b.hobby = "Chess"
RETURN b
```

---

## DELETE — Remove Things

### Delete a relationship

```cypher
MATCH (a:Person {name: "Alice"})-[r:FRIENDS_WITH]->(b:Person {name: "Bob"})
DELETE r
```

### Delete a node (must have no relationships)

```cypher
MATCH (c:Person {name: "Charlie"})
DELETE c
```

> This will **fail** if Charlie has relationships. Use `DETACH DELETE` instead:

### Delete a node + all its relationships

```cypher
MATCH (c:Person {name: "Charlie"})
DETACH DELETE c
```

### Delete everything

```cypher
MATCH (n) DETACH DELETE n
```

---

## MERGE — Create If Not Exists

`CREATE` always makes a new node. `MERGE` checks first — if it exists, it matches it; if not, it creates it.

```cypher
MERGE (a:Person {name: "Alice"})
RETURN a
```

> Run this 10 times — you still get only 1 Alice. With `CREATE` you'd get 10.

This is critical when loading data from CSV or LLM output where duplicates are common.

---

## Live Build Example

Let's build a small company graph from scratch. Run each block in order.

### Reset

```cypher
MATCH (n) DETACH DELETE n
```

### Create the data

```cypher
CREATE
  (alice:Person {name: "Alice", role: "Engineer"}),
  (bob:Person {name: "Bob", role: "Designer"}),
  (carol:Person {name: "Carol", role: "Manager"}),
  (acme:Company {name: "Acme Corp"}),
  (widget:Product {name: "WidgetApp"}),

  (alice)-[:WORKS_AT]->(acme),
  (bob)-[:WORKS_AT]->(acme),
  (carol)-[:MANAGES]->(acme),
  (alice)-[:BUILT]->(widget),
  (bob)-[:DESIGNED]->(widget),
  (carol)-[:APPROVED]->(widget),
  (alice)-[:REPORTS_TO]->(carol),
  (bob)-[:REPORTS_TO]->(carol),
  (alice)-[:COLLABORATES_WITH]->(bob)

RETURN alice, bob, carol, acme, widget
```

> You should see 3 people, 1 company, 1 product — all connected.

### Query it

Who works at Acme?

```cypher
MATCH (p:Person)-[:WORKS_AT]->(c:Company {name: "Acme Corp"})
RETURN p.name, p.role
```

Who built WidgetApp?

```cypher
MATCH (p)-[:BUILT]->(prod:Product {name: "WidgetApp"})
RETURN p.name
```

Who reports to Carol?

```cypher
MATCH (p:Person)-[:REPORTS_TO]->(carol:Person {name: "Carol"})
RETURN p.name
```

What's the full chain from Alice to the product she built, through the company?

```cypher
MATCH path = (alice:Person {name: "Alice"})-[*1..3]-(widget:Product {name: "WidgetApp"})
RETURN path
```

Who are Alice's collaborators' managers?

```cypher
MATCH (alice:Person {name: "Alice"})-[:COLLABORATES_WITH]->(colleague)-[:REPORTS_TO]->(manager)
RETURN alice.name, colleague.name, manager.name
```

> This is a **2-hop query**: Alice → Bob → Carol. You followed two relationships to discover that Alice's collaborator reports to Carol.

---

## Cheat Sheet

```
CREATE (n:Label {key: "value"})         — create a node
CREATE (a)-[:TYPE]->(b)                 — create a relationship
MATCH (n:Label) RETURN n                — find nodes
MATCH (a)-[r]->(b) RETURN a, r, b      — find relationships
WHERE n.key = "value"                   — filter
SET n.key = "new"                       — update
DELETE r                                — delete relationship
DETACH DELETE n                         — delete node + relationships
MERGE (n:Label {key: "value"})          — create if not exists
MATCH (n) DETACH DELETE n               — delete everything
```
