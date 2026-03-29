# Project 12: Recommendation Engine via Knowledge Graph

Build a movie recommendation system powered by a knowledge graph. Users, movies, actors, directors, and genres are connected as a graph, enabling transparent, explainable recommendations through graph traversal and embedding similarity.

## What This Project Does

1. **Build Movie KG** -- Load a movie knowledge graph from JSON into NetworkX. Nodes: Users, Movies, Actors, Directors, Genres. Edges: RATED, HAS_GENRE, HAS_ACTOR, DIRECTED_BY. Print graph stats and most connected entities.
2. **Collaborative Filtering** -- Graph-based collaborative filtering via user similarity. Traverse: User -> rated -> Movie <- rated <- OtherUser. Recommend movies that similar users liked but the target user hasn't seen.
3. **Content-Based Recommendations** -- Recommend movies that share genres, actors, or directors with movies the user liked. Score by number of shared attributes and rank results.
4. **Hybrid Recommendation** -- Combine collaborative filtering, content-based, and embedding similarity into a single weighted scoring system. Return top-K recommendations with fused scores.
5. **Explainable Recommendations** -- Trace the graph paths that led to each recommendation and generate natural-language explanations using LLM.
6. **Visualize Recommendations** -- Interactive pyvis visualization of the recommendation subgraph showing the target user, liked movies, shared attributes, and recommended movies.

## Key Concepts

### Graph-Based Recommendation
- **Collaborative Filtering**: Find users with similar rating patterns by traversing user-movie-user paths
- **Content-Based**: Recommend items sharing attributes with previously liked items
- **Hybrid**: Weighted fusion of multiple recommendation signals

### Why Knowledge Graphs for Recommendations
- **Explainability**: Every recommendation can be traced back to specific graph paths
- **Cold Start**: Content-based paths work even for new users with few ratings
- **Serendipity**: Multi-hop traversals surface unexpected but relevant connections
- **Transparency**: Users can inspect why something was recommended

### Embedding Similarity
- Movie descriptions are embedded via LLM to capture semantic similarity
- Acts as a complementary signal to structural graph features

## Prerequisites

- Python 3.11+
- Dependencies: `networkx`, `numpy`, `pyvis`, `matplotlib`
- Shared LLM layer: `projects/shared/llm_clients.py` (for embeddings and explanations)

## Quick Start

```bash
# Build the movie knowledge graph
python src/01_build_movie_kg.py

# Run collaborative filtering
python src/02_collaborative_filtering.py

# Run content-based recommendations
python src/03_content_based.py

# Run hybrid recommendation engine
python src/04_hybrid_recommendation.py

# Generate explainable recommendations
python src/05_explainable_recs.py

# Visualize the recommendation subgraph
python src/06_visualize_recommendations.py
```

## Graph Schema

```
(User) --[RATED {score}]--> (Movie)
(Movie) --[HAS_GENRE]--> (Genre)
(Movie) --[HAS_ACTOR]--> (Actor)
(Movie) --[DIRECTED_BY]--> (Director)
(Movie) --[SIMILAR_TO]--> (Movie)
```
