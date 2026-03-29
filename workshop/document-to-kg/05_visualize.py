"""Hour 2 — Script 05: Visualize the Knowledge Graph"""
from pathlib import Path
from neo4j import GraphDatabase
from pyvis.network import Network

HERE = Path(__file__).parent
COLORS = {
    "Person": "#FF6B6B", "Organization": "#4ECDC4", "Technology": "#45B7D1",
    "Model": "#96CEB4", "Method": "#FFEAA7", "Database": "#DDA0DD",
    "Framework": "#F0E68C",
}

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "workshop2024"))
net = Network(height="700px", width="100%", directed=True, bgcolor="#ffffff")
net.barnes_hut(gravity=-3000)

with driver.session() as s:
    for r in s.run("MATCH (n:Entity) RETURN n.name AS name, n.type AS type, n.description AS desc"):
        color = COLORS.get(r["type"], "#DFE6E9")
        net.add_node(r["name"], label=r["name"], color=color,
                     title=f"{r['type']}: {r['desc']}", size=20)

    for r in s.run("MATCH (a)-[r]->(b) RETURN a.name AS src, b.name AS tgt, r.type AS type"):
        net.add_edge(r["src"], r["tgt"], label=r["type"], color="#B2BEC3")

driver.close()

(HERE / "output").mkdir(exist_ok=True)
output = str(HERE / "output" / "knowledge_graph.html")
net.save_graph(output)
print(f"Interactive graph saved to: {output}")
print("Open it in your browser!")
