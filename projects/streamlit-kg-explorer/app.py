"""Streamlit KG Explorer — Interactive knowledge graph exploration app.

Run with: streamlit run app.py
"""

import json
import sys
import tempfile
from pathlib import Path
from collections import Counter

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KG Explorer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ───────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = None
if "selected_node" not in st.session_state:
    st.session_state.selected_node = None
if "navigation_history" not in st.session_state:
    st.session_state.navigation_history = []


# ── Graph loading functions ──────────────────────────────────────────────

def load_graphml(file_content: bytes) -> nx.Graph:
    """Load a GraphML file into NetworkX."""
    with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as tmp:
        tmp.write(file_content)
        tmp.flush()
        G = nx.read_graphml(tmp.name)
    return G


def load_json_graph(file_content: bytes) -> nx.Graph:
    """Load a JSON graph file with nodes and edges."""
    data = json.loads(file_content.decode("utf-8"))
    G = nx.DiGraph()

    for node in data.get("nodes", []):
        node_id = node.pop("id", None) or node.pop("name", str(len(G)))
        G.add_node(node_id, **node)

    for edge in data.get("edges", data.get("links", [])):
        source = edge.pop("source", None)
        target = edge.pop("target", None)
        if source and target:
            G.add_edge(source, target, **edge)

    return G


def load_from_neo4j(uri: str, user: str, password: str, limit: int = 500) -> nx.Graph:
    """Load a subgraph from Neo4j."""
    from neo4j import GraphDatabase

    G = nx.DiGraph()
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        result = session.run(
            f"MATCH (a)-[r]->(b) RETURN a, r, b LIMIT {limit}"
        )
        for record in result:
            a = record["a"]
            b = record["b"]
            r = record["r"]

            a_id = str(a.element_id)
            b_id = str(b.element_id)

            a_props = dict(a)
            a_props["neo4j_labels"] = ", ".join(a.labels)
            a_props["label"] = a_props.get("name", a_props.get("title", a_id))

            b_props = dict(b)
            b_props["neo4j_labels"] = ", ".join(b.labels)
            b_props["label"] = b_props.get("name", b_props.get("title", b_id))

            G.add_node(a_id, **{k: str(v) for k, v in a_props.items()})
            G.add_node(b_id, **{k: str(v) for k, v in b_props.items()})
            G.add_edge(a_id, b_id, type=r.type, **{k: str(v) for k, v in dict(r).items()})

    driver.close()
    return G


# ── Sidebar: Data source ────────────────────────────────────────────────

st.sidebar.title("KG Explorer")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Data Source",
    ["Upload GraphML", "Upload JSON", "Connect to Neo4j"],
)

if data_source == "Upload GraphML":
    uploaded = st.sidebar.file_uploader("Upload a .graphml file", type=["graphml"])
    if uploaded is not None:
        try:
            st.session_state.graph = load_graphml(uploaded.read())
            st.sidebar.success(f"Loaded graph: {st.session_state.graph.number_of_nodes()} nodes, {st.session_state.graph.number_of_edges()} edges")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

elif data_source == "Upload JSON":
    uploaded = st.sidebar.file_uploader("Upload a .json file", type=["json"])
    if uploaded is not None:
        try:
            st.session_state.graph = load_json_graph(uploaded.read())
            st.sidebar.success(f"Loaded graph: {st.session_state.graph.number_of_nodes()} nodes, {st.session_state.graph.number_of_edges()} edges")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

elif data_source == "Connect to Neo4j":
    neo4j_uri = st.sidebar.text_input("Neo4j URI", value="bolt://localhost:7687")
    neo4j_user = st.sidebar.text_input("Username", value="neo4j")
    neo4j_password = st.sidebar.text_input("Password", type="password")
    neo4j_limit = st.sidebar.number_input("Max relationships", value=500, min_value=10, max_value=10000)

    if st.sidebar.button("Connect"):
        try:
            st.session_state.graph = load_from_neo4j(neo4j_uri, neo4j_user, neo4j_password, neo4j_limit)
            st.sidebar.success(f"Connected! Loaded {st.session_state.graph.number_of_nodes()} nodes, {st.session_state.graph.number_of_edges()} edges")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")

# ── Main content ─────────────────────────────────────────────────────────

G = st.session_state.graph

if G is None:
    st.title("Knowledge Graph Explorer")
    st.markdown("""
    Welcome to the KG Explorer. Upload a graph file or connect to Neo4j using the sidebar.

    **Supported formats:**
    - **GraphML** — Export from NetworkX with `nx.write_graphml(G, 'graph.graphml')`
    - **JSON** — `{"nodes": [{"id": "a", ...}], "edges": [{"source": "a", "target": "b", ...}]}`
    - **Neo4j** — Connect to a running Neo4j database
    """)
    st.stop()


# ── Tabs ─────────────────────────────────────────────────────────────────

tab_explorer, tab_graph, tab_stats, tab_query = st.tabs([
    "Explorer", "Graph View", "Statistics", "Query"
])


# ── Tab 1: Explorer ─────────────────────────────────────────────────────

with tab_explorer:
    st.header("Entity Explorer")

    # Search bar
    all_nodes = sorted(G.nodes())
    node_labels = {n: G.nodes[n].get("label", G.nodes[n].get("title", n)) for n in all_nodes}
    display_options = [f"{node_labels[n]}  ({n})" if node_labels[n] != n else n for n in all_nodes]

    col_search, col_nav = st.columns([3, 1])
    with col_search:
        search_term = st.text_input("Search entities", placeholder="Type to search...")
    with col_nav:
        if st.session_state.navigation_history:
            if st.button("Back"):
                if len(st.session_state.navigation_history) > 1:
                    st.session_state.navigation_history.pop()
                    st.session_state.selected_node = st.session_state.navigation_history[-1]
                    st.rerun()

    # Filter nodes by search term
    if search_term:
        matching = [
            n for n in all_nodes
            if search_term.lower() in n.lower()
            or search_term.lower() in str(node_labels.get(n, "")).lower()
            or search_term.lower() in str(G.nodes[n]).lower()
        ]
    else:
        matching = all_nodes[:50]  # Show first 50 if no search

    # Node selector
    if matching:
        selected = st.selectbox(
            f"Select entity ({len(matching)} matches)",
            matching,
            format_func=lambda n: node_labels.get(n, n),
        )
        if selected:
            st.session_state.selected_node = selected
            if not st.session_state.navigation_history or st.session_state.navigation_history[-1] != selected:
                st.session_state.navigation_history.append(selected)

    # Show selected node details
    if st.session_state.selected_node and st.session_state.selected_node in G:
        node = st.session_state.selected_node
        data = G.nodes[node]

        st.subheader(f"{node_labels.get(node, node)}")

        # Properties
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Properties:**")
            for key, value in data.items():
                st.markdown(f"- **{key}**: {value}")

        with col2:
            # Degree info
            if G.is_directed():
                st.metric("In-degree", G.in_degree(node))
                st.metric("Out-degree", G.out_degree(node))
            else:
                st.metric("Degree", G.degree(node))

        # Neighbors and relationships
        st.markdown("---")
        st.markdown("**Relationships:**")

        if G.is_directed():
            # Outgoing edges
            outgoing = list(G.successors(node))
            if outgoing:
                st.markdown(f"**Outgoing ({len(outgoing)}):**")
                for neighbor in outgoing:
                    edge_data = G[node][neighbor]
                    edge_type = edge_data.get("edge_type", edge_data.get("type", ""))
                    label = node_labels.get(neighbor, neighbor)
                    edge_label = f" --[{edge_type}]--> " if edge_type else " --> "

                    if st.button(f"{edge_label} {label}", key=f"out_{node}_{neighbor}"):
                        st.session_state.selected_node = neighbor
                        st.session_state.navigation_history.append(neighbor)
                        st.rerun()

            # Incoming edges
            incoming = list(G.predecessors(node))
            if incoming:
                st.markdown(f"**Incoming ({len(incoming)}):**")
                for neighbor in incoming:
                    edge_data = G[neighbor][node]
                    edge_type = edge_data.get("edge_type", edge_data.get("type", ""))
                    label = node_labels.get(neighbor, neighbor)
                    edge_label = f" <--[{edge_type}]-- " if edge_type else " <-- "

                    if st.button(f"{label} {edge_label}", key=f"in_{neighbor}_{node}"):
                        st.session_state.selected_node = neighbor
                        st.session_state.navigation_history.append(neighbor)
                        st.rerun()
        else:
            neighbors = list(G.neighbors(node))
            if neighbors:
                st.markdown(f"**Neighbors ({len(neighbors)}):**")
                for neighbor in neighbors:
                    label = node_labels.get(neighbor, neighbor)
                    if st.button(f"{label}", key=f"nb_{node}_{neighbor}"):
                        st.session_state.selected_node = neighbor
                        st.session_state.navigation_history.append(neighbor)
                        st.rerun()


# ── Tab 2: Graph View ───────────────────────────────────────────────────

with tab_graph:
    st.header("Graph Visualization")

    max_nodes = st.slider("Max nodes to display", 10, 500, min(100, G.number_of_nodes()))

    # Subsample if needed
    if G.number_of_nodes() > max_nodes:
        # Take highest-degree nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        subgraph = G.subgraph(top_nodes)
        st.info(f"Showing top {max_nodes} nodes by degree (of {G.number_of_nodes()} total)")
    else:
        subgraph = G

    # Build pyvis visualization
    from pyvis.network import Network

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=G.is_directed(),
        notebook=False,
    )
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 150,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100}
        }
    }
    """)

    # Add nodes
    for node, data in subgraph.nodes(data=True):
        label = data.get("label", data.get("title", node))
        node_type = data.get("node_type", "default")
        size = 20 if node_type == "NOTE" else 10
        color = "#4e79a7" if node_type == "NOTE" else "#e15759" if node_type == "CONCEPT" else "#76b7b2"
        net.add_node(node, label=str(label), size=size, color=color, title=str(data))

    # Add edges
    for source, target, data in subgraph.edges(data=True):
        edge_type = data.get("edge_type", data.get("type", ""))
        net.add_edge(source, target, title=edge_type)

    # Render
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as tmp:
        net.save_graph(tmp.name)
        html_content = Path(tmp.name).read_text()
        st.components.html(html_content, height=650, scrolling=True)


# ── Tab 3: Statistics ────────────────────────────────────────────────────

with tab_stats:
    st.header("Graph Statistics")

    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", G.number_of_nodes())
    col2.metric("Edges", G.number_of_edges())
    density = nx.density(G)
    col3.metric("Density", f"{density:.4f}")
    if not G.is_directed():
        components = nx.number_connected_components(G)
    else:
        components = nx.number_weakly_connected_components(G)
    col4.metric("Components", components)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # Node type distribution
        node_types = Counter(
            data.get("node_type", data.get("neo4j_labels", "unknown"))
            for _, data in G.nodes(data=True)
        )
        if len(node_types) > 1 or list(node_types.keys()) != ["unknown"]:
            fig_types = px.bar(
                x=list(node_types.keys()),
                y=list(node_types.values()),
                labels={"x": "Node Type", "y": "Count"},
                title="Node Type Distribution",
            )
            st.plotly_chart(fig_types, use_container_width=True)

        # Edge type distribution
        edge_types = Counter(
            data.get("edge_type", data.get("type", "unknown"))
            for _, _, data in G.edges(data=True)
        )
        if len(edge_types) > 1 or list(edge_types.keys()) != ["unknown"]:
            fig_edges = px.bar(
                x=list(edge_types.keys()),
                y=list(edge_types.values()),
                labels={"x": "Edge Type", "y": "Count"},
                title="Edge Type Distribution",
            )
            st.plotly_chart(fig_edges, use_container_width=True)

    with col_right:
        # Degree distribution
        degrees = [d for _, d in G.degree()]
        fig_degree = px.histogram(
            x=degrees,
            nbins=min(30, max(degrees) + 1) if degrees else 10,
            labels={"x": "Degree", "y": "Count"},
            title="Degree Distribution",
        )
        st.plotly_chart(fig_degree, use_container_width=True)

        # Top entities by centrality
        st.markdown("**Top Entities by Betweenness Centrality:**")
        try:
            centrality = nx.betweenness_centrality(G)
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            for node_id, score in top_central:
                label = G.nodes[node_id].get("label", G.nodes[node_id].get("title", node_id))
                st.markdown(f"- **{label}**: {score:.4f}")
        except Exception:
            st.info("Centrality computation not available for this graph type.")

    # Community detection
    st.markdown("---")
    st.subheader("Community Detection")
    try:
        undirected = G.to_undirected() if G.is_directed() else G
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(undirected))
        st.markdown(f"Found **{len(communities)}** communities:")
        for i, community in enumerate(communities[:10]):
            members = sorted(community)[:10]
            labels = [G.nodes[n].get("label", G.nodes[n].get("title", n)) for n in members]
            suffix = f"... (+{len(community) - 10} more)" if len(community) > 10 else ""
            st.markdown(f"- **Community {i + 1}** ({len(community)} members): {', '.join(str(l) for l in labels)}{suffix}")
    except Exception as e:
        st.info(f"Community detection unavailable: {e}")


# ── Tab 4: Query ─────────────────────────────────────────────────────────

with tab_query:
    st.header("Natural Language Query")
    st.markdown("Ask questions about your knowledge graph. Uses an LLM to answer based on graph context.")

    question = st.text_input("Ask a question about the graph", placeholder="e.g., What are the most connected entities?")

    col_provider, col_model = st.columns(2)
    with col_provider:
        provider = st.selectbox("LLM Provider", ["openai", "anthropic", "ollama"])
    with col_model:
        model = st.text_input("Model (leave empty for default)", value="")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            try:
                from shared.llm_clients import chat_completion

                # Build graph context
                context_parts = []

                # Graph summary
                context_parts.append(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

                # Node types
                node_types = Counter(
                    data.get("node_type", "unknown") for _, data in G.nodes(data=True)
                )
                context_parts.append(f"Node types: {dict(node_types)}")

                # Edge types
                edge_types = Counter(
                    data.get("edge_type", data.get("type", "unknown"))
                    for _, _, data in G.edges(data=True)
                )
                context_parts.append(f"Edge types: {dict(edge_types)}")

                # Top nodes by degree
                top_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:20]
                top_nodes_str = "\n".join(
                    f"  - {G.nodes[n].get('label', n)} (degree: {d})" for n, d in top_by_degree
                )
                context_parts.append(f"Top nodes by degree:\n{top_nodes_str}")

                # Sample edges
                sample_edges = list(G.edges(data=True))[:30]
                edges_str = "\n".join(
                    f"  - {G.nodes[s].get('label', s)} --[{d.get('edge_type', d.get('type', ''))}]--> {G.nodes[t].get('label', t)}"
                    for s, t, d in sample_edges
                )
                context_parts.append(f"Sample relationships:\n{edges_str}")

                context = "\n\n".join(context_parts)

                system = """You are a knowledge graph analyst. Answer questions about the
graph based on the provided context. Be specific and reference actual
entities and relationships from the graph."""

                prompt = f"""Here is information about a knowledge graph:

{context}

Question: {question}"""

                answer = chat_completion(
                    prompt=prompt,
                    system=system,
                    provider=provider,
                    model=model if model else None,
                )

                st.markdown("**Answer:**")
                st.markdown(answer)

            except ImportError:
                st.error("Could not import shared.llm_clients. Make sure the shared module is available.")
            except Exception as e:
                st.error(f"Error: {e}")
