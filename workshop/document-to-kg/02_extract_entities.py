"""Hour 2 — Script 02: Extract Entities from the Paper"""
import json, sys
from pathlib import Path
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "projects"))

from shared.llm_clients import chat_completion_structured


class Entity(BaseModel):
    name: str = Field(description="Entity name")
    type: str = Field(description="PERSON, ORGANIZATION, TECHNOLOGY, MODEL, METHOD, DATABASE, or FRAMEWORK")
    description: str = Field(description="One-line description from the text")

class EntityList(BaseModel):
    entities: list[Entity] = Field(default_factory=list)


paper = (HERE / "data" / "ai_agents_paper.txt").read_text()
print(f"Loaded paper: {len(paper)} chars\n")

print("Extracting entities with LLM...")
result = chat_completion_structured(
    prompt=f"Extract all entities from this research paper:\n\n{paper}",
    output_schema=EntityList,
    system="Extract entities. Types: PERSON, ORGANIZATION, TECHNOLOGY, MODEL, METHOD, DATABASE, FRAMEWORK.",
)

print(f"\nFound {len(result.entities)} entities:\n")
for e in result.entities:
    print(f"  [{e.type:15}] {e.name}")
    print(f"                  {e.description}\n")

(HERE / "output").mkdir(exist_ok=True)
entities = [e.model_dump() for e in result.entities]
(HERE / "output" / "entities.json").write_text(json.dumps(entities, indent=2))
print(f"Saved {len(entities)} entities to output/entities.json")
