"""Hour 2 — Script 03: Extract Relationships Between Entities"""
import json, sys
from pathlib import Path
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "projects"))

from shared.llm_clients import chat_completion_structured


class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description="e.g. AUTHORED, USES, STORED_IN, EVALUATED_ON, PART_OF")
    description: str = Field(description="One-line description")

class RelationshipList(BaseModel):
    relationships: list[Relationship] = Field(default_factory=list)


paper = (HERE / "data" / "ai_agents_paper.txt").read_text()
entities = json.loads((HERE / "output" / "entities.json").read_text())
entity_names = [e["name"] for e in entities]
print(f"Paper: {len(paper)} chars | Entities: {len(entity_names)}\n")

print("Extracting relationships with LLM...")
result = chat_completion_structured(
    prompt=f"Known entities:\n{chr(10).join('- ' + n for n in entity_names)}\n\nText:\n{paper}",
    output_schema=RelationshipList,
    system="Extract relationships between the given entities from the text.",
)

print(f"\nFound {len(result.relationships)} relationships:\n")
for r in result.relationships:
    print(f"  {r.source}  --[{r.type}]-->  {r.target}")
    print(f"    {r.description}\n")

rels = [r.model_dump() for r in result.relationships]
(HERE / "output" / "relations.json").write_text(json.dumps(rels, indent=2))
print(f"Saved {len(rels)} relationships to output/relations.json")
