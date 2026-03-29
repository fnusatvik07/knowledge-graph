# Beyond Text: Multimodal Knowledge Graphs

Most knowledge graphs are built from text. But the real world is multimodal -- knowledge is embedded in images, tables, code, audio, video, and spatial data. Multimodal knowledge graphs integrate structured knowledge from multiple modalities into a unified graph representation.

## Why Multimodal KGs?

Consider a biomedical knowledge graph. Text-only extraction misses:
- **Molecular structures** depicted in figures
- **Experimental results** in tables
- **Code** implementing algorithms described in papers
- **Clinical images** (X-rays, MRIs) annotated with findings

A multimodal KG links all of these into a single graph where relationships cross modality boundaries.

## Image Knowledge Graphs (Scene Graphs)

Scene graphs represent the contents of images as structured graphs: objects as nodes, spatial/semantic relationships as edges.

### Structure

```
Image: "A dog sitting on a couch next to a person reading a book"

Scene Graph:
  (dog) --[sitting_on]--> (couch)
  (person) --[sitting_on]--> (couch)
  (person) --[reading]--> (book)
  (dog) --[next_to]--> (person)
```

### Visual Genome Dataset

**URL**: https://homes.cs.washington.edu/~ranjay/visualgenome/
**Size**: 108K images, 3.8M object instances, 2.3M relationships
**Format**: JSON

```python
import json
import requests

# Download scene graphs
# https://homes.cs.washington.edu/~ranjay/visualgenome/api.html

def load_scene_graphs(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)

# Example scene graph structure
scene_graph = {
    "image_id": 1,
    "objects": [
        {"object_id": 1, "names": ["dog"], "x": 100, "y": 200, "w": 50, "h": 80},
        {"object_id": 2, "names": ["couch"], "x": 50, "y": 150, "w": 200, "h": 150},
    ],
    "relationships": [
        {"relationship_id": 1, "subject_id": 1, "object_id": 2, "predicate": "sitting on"}
    ]
}
```

### Generating Scene Graphs with Vision LLMs

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import base64

class SceneObject(BaseModel):
    name: str
    attributes: list[str] = Field(default_factory=list)

class SceneRelationship(BaseModel):
    subject: str
    predicate: str
    object: str

class SceneGraph(BaseModel):
    objects: list[SceneObject]
    relationships: list[SceneRelationship]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(SceneGraph)

def extract_scene_graph(image_path: str) -> SceneGraph:
    """Extract a scene graph from an image using a vision LLM."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    message = HumanMessage(content=[
        {"type": "text", "text": "Extract all objects and their relationships from this image as a scene graph."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    ])

    return structured_llm.invoke([message])
```

> **LangChain multimodal**: https://python.langchain.com/docs/how_to/multimodal_inputs/

## Table Knowledge Graphs

Tables encode structured knowledge that can be directly converted to graph triples.

### Table-to-Graph Conversion

```python
import pandas as pd

def table_to_triples(df: pd.DataFrame, subject_col: str, mappings: dict) -> list[tuple]:
    """Convert a DataFrame to knowledge graph triples.

    Args:
        df: Input DataFrame
        subject_col: Column to use as the subject entity
        mappings: Dict mapping column names to relationship types
                  e.g., {"company": "WORKS_AT", "city": "LOCATED_IN"}
    """
    triples = []
    for _, row in df.iterrows():
        subject = str(row[subject_col])
        for col, rel_type in mappings.items():
            if pd.notna(row[col]):
                triples.append((subject, rel_type, str(row[col])))
    return triples

# Example
df = pd.DataFrame({
    "person": ["Alice", "Bob", "Charlie"],
    "company": ["Google", "Meta", "Google"],
    "city": ["Mountain View", "Menlo Park", "New York"],
})

triples = table_to_triples(df, "person", {"company": "WORKS_AT", "city": "LIVES_IN"})
for s, r, o in triples:
    print(f"  ({s}) --[{r}]--> ({o})")
```

### TableQA with Knowledge Graphs

Combine table understanding with graph reasoning:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0)

table_kg_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data analyst. Given a table and a knowledge graph schema,
extract entities and relationships from the table and express them as graph triples.

Knowledge graph schema:
{schema}

Return triples as: (subject, RELATIONSHIP_TYPE, object)"""),
    ("human", "Table:\n{table}\n\nExtract all triples.")
])
```

> **LangChain prompt templates**: https://python.langchain.com/docs/how_to/prompts_composition/

## Code Knowledge Graphs (AST Graphs)

Source code has rich structure that maps naturally to graphs. Abstract Syntax Trees (ASTs), call graphs, and dependency graphs are all forms of code knowledge graphs.

### AST to Knowledge Graph

```python
import ast

def code_to_kg_triples(source_code: str) -> list[tuple]:
    """Extract knowledge graph triples from Python source code."""
    tree = ast.parse(source_code)
    triples = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            triples.append((node.name, "IS_A", "Class"))
            for base in node.bases:
                if isinstance(base, ast.Name):
                    triples.append((node.name, "INHERITS_FROM", base.id))
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    triples.append((node.name, "HAS_METHOD", item.name))

        elif isinstance(node, ast.FunctionDef):
            triples.append((node.name, "IS_A", "Function"))
            for arg in node.args.args:
                if arg.arg != "self":
                    triples.append((node.name, "HAS_PARAMETER", arg.arg))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                triples.append(("module", "IMPORTS", alias.name))

    return triples

# Example
code = '''
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof"

    def fetch(self, item):
        pass
'''

triples = code_to_kg_triples(code)
for s, r, o in triples:
    print(f"  ({s}) --[{r}]--> ({o})")
# (Animal) --[IS_A]--> (Class)
# (Animal) --[HAS_METHOD]--> (speak)
# (Dog) --[IS_A]--> (Class)
# (Dog) --[INHERITS_FROM]--> (Animal)
# (Dog) --[HAS_METHOD]--> (speak)
# (Dog) --[HAS_METHOD]--> (fetch)
# (fetch) --[HAS_PARAMETER]--> (item)
```

## Audio and Transcript Knowledge Graphs

Audio content (podcasts, meetings, lectures) can be transcribed and then processed through standard text extraction pipelines.

```python
# Pipeline: Audio -> Transcript -> Entity Extraction -> Knowledge Graph
# Step 1: Transcribe with Whisper (via LangChain or directly)
# Step 2: Extract entities with LangChain structured output
# Step 3: Build graph

from langchain_openai import ChatOpenAI

# After transcription, use standard extraction
# See: ../02-kg-construction/04-llm-extraction-patterns.md
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(ExtractionResult)
```

> **LangChain structured output**: https://python.langchain.com/docs/how_to/structured_output/

## Cross-Modal Embeddings

To connect entities across modalities, use cross-modal embedding models that map different data types into a shared vector space.

### CLIP for Image-Text Alignment

```python
# Map image entities and text entities to the same embedding space
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Text embeddings for graph entities
entity_names = ["golden retriever", "modern sofa", "hardcover book"]
text_inputs = processor(text=entity_names, return_tensors="pt", padding=True)
text_embeddings = model.get_text_features(**text_inputs)

# Image embeddings for scene graph objects
# image_inputs = processor(images=[img1, img2], return_tensors="pt")
# image_embeddings = model.get_image_features(**image_inputs)

# Cosine similarity for entity alignment across modalities
# similarity = torch.cosine_similarity(text_embeddings, image_embeddings)
```

## Real-World Multimodal KGs

### Google Knowledge Graph

Google's Knowledge Graph is perhaps the best-known multimodal KG:
- **Text**: Entity descriptions, Wikipedia summaries
- **Images**: Entity photos, logos, maps
- **Structured data**: Infobox facts (founding date, headquarters, etc.)
- **Relationships**: "People also search for", "Related entities"

### Biomedical Multimodal KGs

Biomedical research increasingly combines:
- **Literature** (PubMed abstracts) -> text-based entities and relations
- **Molecular structures** (PDB, ChEMBL) -> 3D structure graphs
- **Clinical images** (radiology, pathology) -> image-based findings
- **Genomic data** (Gene Ontology) -> biological process graphs

```python
# Example: Link a drug entity to both its text description and molecular structure
drug_node = {
    "name": "Aspirin",
    "type": "Drug",
    "text_description": "Nonsteroidal anti-inflammatory drug...",
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # Molecular structure
    "image_url": "https://pubchem.ncbi.nlm.nih.gov/image/...",
    "pubchem_id": "2244",
}
```

## The Future: Video and Spatial KGs

### Video Knowledge Graphs

Video extends scene graphs through time: tracking objects, actions, and events across frames.

```
Frame 1: (person_A) --[enters]--> (room)
Frame 2: (person_A) --[picks_up]--> (phone)
Frame 3: (person_A) --[calls]--> (person_B)

Temporal KG:
  (person_A) --[enters {t=0s}]--> (room)
  (person_A) --[picks_up {t=2s}]--> (phone)
  (person_A) --[calls {t=5s}]--> (person_B)
```

### Spatial Knowledge Graphs

Spatial KGs encode geographic relationships: containment, adjacency, proximity, and direction.

```
(Central Park) --[CONTAINED_IN]--> (Manhattan)
(Manhattan) --[PART_OF]--> (New York City)
(Empire State Building) --[NEAR {distance_km: 0.8}]--> (Central Park)
(Brooklyn Bridge) --[CONNECTS]--> (Manhattan)
(Brooklyn Bridge) --[CONNECTS]--> (Brooklyn)
```

## Building Multimodal KGs: A Framework

```python
from enum import Enum

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"
    AUDIO = "audio"

class MultimodalEntity:
    def __init__(self, name: str, entity_type: str):
        self.name = name
        self.entity_type = entity_type
        self.modality_data: dict[Modality, any] = {}
        self.embeddings: dict[Modality, list[float]] = {}

    def add_modality(self, modality: Modality, data: any, embedding: list[float] = None):
        self.modality_data[modality] = data
        if embedding:
            self.embeddings[modality] = embedding

    def cross_modal_similarity(self, other: "MultimodalEntity") -> float:
        """Compute similarity across shared modalities."""
        import numpy as np
        shared = set(self.embeddings.keys()) & set(other.embeddings.keys())
        if not shared:
            return 0.0
        similarities = []
        for modality in shared:
            a = np.array(self.embeddings[modality])
            b = np.array(other.embeddings[modality])
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            similarities.append(sim)
        return sum(similarities) / len(similarities)
```

## Next Steps

- [02 - KG Versioning and CI/CD](./02-kg-versioning-and-cicd.md) -- managing the lifecycle of your KG
- [LLM Extraction Patterns](../02-kg-construction/04-llm-extraction-patterns.md) -- text extraction as a starting point
- [Graph Embeddings](../05-advanced-topics/01-graph-embeddings.md) -- embedding methods for graphs
