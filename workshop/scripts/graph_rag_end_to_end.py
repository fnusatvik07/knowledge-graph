"""
COMPLETE GRAPH RAG PIPELINE — One Self-Contained Script
========================================================
Document → LLM extracts KG → Load into Neo4j → User asks question →
LLM generates Cypher → Query Neo4j → LLM answers with graph context

Prerequisites:
  pip install langchain-openai neo4j pydantic python-dotenv
  docker compose -f workshop/docker-compose.yml up -d
"""

from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ═══════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════
HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / ".env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "workshop2024")
DOCUMENT_PATH = HERE / "document-to-kg" / "data" / "ai_agents_paper.txt"


# ═══════════════════════════════════════════════════════════════════
# HELPER: call the LLM
# ═══════════════════════════════════════════════════════════════════
def ask_llm(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Simple LLM call — send a prompt, get a string back."""
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])
    return response.content


# ═══════════════════════════════════════════════════════════════════
# PART 1: EXTRACT KNOWLEDGE GRAPH FROM DOCUMENT
# ═══════════════════════════════════════════════════════════════════

class Entity(BaseModel):
    name: str = Field(description="Entity name")
    type: str = Field(description="PERSON, ORGANIZATION, TECHNOLOGY, MODEL, METHOD, DATABASE, or FRAMEWORK")
    description: str = Field(description="One-line description")

class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description="Relationship type like AUTHORED, USES, BUILT, STORED_IN, PART_OF")
    description: str = Field(description="One-line description")

class KnowledgeGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


def extract_knowledge_graph(text: str) -> KnowledgeGraph:
    """Send document to LLM → get back structured entities + relationships."""
    print("\n   Sending document to LLM for extraction...")

    structured_llm = llm.with_structured_output(KnowledgeGraph)

    kg = structured_llm.invoke([
        SystemMessage(content="""You are a knowledge graph extraction expert.
Extract entities (PERSON, ORGANIZATION, TECHNOLOGY, MODEL, METHOD, DATABASE, FRAMEWORK)
and relationships between them from the given text.
Only extract what is explicitly stated in the text."""),
        HumanMessage(content=f"Extract all entities and relationships from this document:\n\n{text}"),
    ])

    print(f"   Found {len(kg.entities)} entities and {len(kg.relationships)} relationships")
    return kg


# ═══════════════════════════════════════════════════════════════════
# PART 2: LOAD INTO NEO4J
# ═══════════════════════════════════════════════════════════════════

def load_into_neo4j(kg: KnowledgeGraph):
    """Take extracted KG → run Cypher CREATE queries → store in Neo4j."""
    print("\n   Loading into Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    with driver.session() as s:
        # Clear old data
        s.run("MATCH (n) DETACH DELETE n")

        # Create nodes
        for e in kg.entities:
            s.run(
                "CREATE (n:Entity {name: $name, type: $type, description: $desc})",
                name=e.name, type=e.type, desc=e.description,
            )

        # Create relationships
        for r in kg.relationships:
            s.run(
                """MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
                   CREATE (a)-[:RELATES_TO {type: $rel_type, description: $desc}]->(b)""",
                src=r.source, tgt=r.target, rel_type=r.type, desc=r.description,
            )

        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        edges = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    driver.close()
    print(f"   Loaded: {nodes} nodes, {edges} relationships")
    print(f"   View graph: http://localhost:7474")
    print(f"   Run: MATCH (n)-[r]->(m) RETURN n, r, m")


# ═══════════════════════════════════════════════════════════════════
# PART 3: GRAPH RAG — ANSWER QUESTIONS
# ═══════════════════════════════════════════════════════════════════

def graph_rag_query(question: str) -> str:
    """
    User question → LLM generates Cypher → Query Neo4j → LLM answers

    TWO LLM calls:
      1. Question → Cypher query  (text-to-cypher)
      2. Graph results + Question → Natural language answer
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    # First, get the actual entity names from the graph so LLM knows what exists
    with driver.session() as s:
        entity_names = [r["name"] for r in s.run("MATCH (n:Entity) RETURN n.name AS name")]

    # --- LLM CALL 1: Generate Cypher from the question ---
    cypher = ask_llm(
        prompt=f"Convert this question to a Cypher query:\n\n{question}",
        system=f"""You are a Neo4j Cypher expert.
The graph schema:
  - Nodes: label :Entity, properties: name (string), type (string), description (string)
  - Relationships: type :RELATES_TO, properties: type (string), description (string)

Existing entity names in the graph: {entity_names}

IMPORTANT: Always use toLower() and CONTAINS for name matching. Never match exact names.
Return ONLY the Cypher query. No markdown, no explanation.
Example: MATCH (a:Entity)-[r]->(b:Entity) WHERE toLower(a.name) CONTAINS 'rdf' RETURN a.name, r.type, r.description, b.name""",
    )

    cypher = cypher.strip().replace("```cypher", "").replace("```", "").strip()
    print(f"   Cypher: {cypher}")

    # --- Run Cypher on Neo4j ---
    try:
        with driver.session() as s:
            rows = [dict(r) for r in s.run(cypher)]
    except Exception:
        rows = []

    # Fallback: if Cypher returned nothing, grab all triples
    if not rows:
        print(f"   No targeted results, fetching full graph...")
        with driver.session() as s:
            rows = [dict(r) for r in s.run(
                "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name AS source, r.type AS rel, b.name AS target"
            )]

    driver.close()
    print(f"   Got {len(rows)} results from Neo4j")

    if not rows:
        return "No data in the knowledge graph."

    graph_context = "\n".join([str(row) for row in rows])

    # --- LLM CALL 2: Answer using graph results ---
    answer = ask_llm(
        prompt=f"Knowledge Graph results:\n{graph_context}\n\nQuestion: {question}",
        system="Answer the question using ONLY the knowledge graph results. Be specific and cite the relationships you found.",
    )

    return answer


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GRAPH RAG — End to End Pipeline")
    print("=" * 60)

    # --- Step 1: Read document ---
    print("\nSTEP 1: Reading document...")
    document = DOCUMENT_PATH.read_text()
    print(f"   Loaded: {len(document)} characters")

    # --- Step 2: Extract KG ---
    print("\nSTEP 2: Extracting Knowledge Graph with LLM...")
    kg = extract_knowledge_graph(document)

    print("\n   Entities:")
    for e in kg.entities:
        print(f"     [{e.type}] {e.name}")

    print("\n   Relationships:")
    for r in kg.relationships:
        print(f"     {r.source} --[{r.type}]--> {r.target}")

    # --- Step 3: Load into Neo4j ---
    print("\nSTEP 3: Loading into Neo4j...")
    load_into_neo4j(kg)

    # --- Step 4: Graph RAG ---
    print("\n" + "=" * 60)
    print("STEP 4: Graph RAG — Ask Questions!")
    print("=" * 60)

    questions = [
        "What technologies does the RDF pipeline use?",
        "Who are the authors and what organization are they from?",
        "What is the relationship between BGE-m3 and the RAG2 baseline?",
        "Compare LPG and RDF — which performed better?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = graph_rag_query(q)
        print(f"\nA: {answer}")
        print("-" * 60)

    # --- Interactive mode ---
    print("\n\nAsk your own questions (type 'quit' to exit):\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in ("quit", "exit", "q", ""):
            break
        print(f"\nQ: {q}")
        answer = graph_rag_query(q)
        print(f"\nA: {answer}\n")
