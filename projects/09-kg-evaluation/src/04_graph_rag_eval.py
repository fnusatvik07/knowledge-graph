"""Evaluate a Graph RAG system using Neo4j for knowledge graph retrieval.

Builds a graph RAG from the gold standard documents (entity extraction + Neo4j),
runs the same evaluation questions as the vector RAG, and scores with identical
RAGAS-inspired metrics for direct comparison.

LangChain Neo4j: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
LangChain structured output: https://python.langchain.com/docs/how_to/structured_output/

Usage:
    python src/04_graph_rag_eval.py
    python src/04_graph_rag_eval.py --provider openai --build-graph
"""

import sys
import json
import argparse
from pathlib import Path

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import BaseModel, Field
from neo4j import GraphDatabase

from shared.llm_clients import (
    chat_completion,
    chat_completion_structured,
    get_embedding,
)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
GOLD_STANDARD_PATH = PROJECT_DIR / "data" / "gold_standard.json"
EVAL_QUESTIONS_PATH = PROJECT_DIR / "data" / "eval_questions.json"
OUTPUT_DIR = PROJECT_DIR / "output"

# ── Neo4j settings ───────────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


# ── Pydantic schemas ─────────────────────────────────────────────────────

class ExtractedEntity(BaseModel):
    name: str
    type: str


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    type: str
    description: str = ""


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


class QueryEntities(BaseModel):
    """Entities to look up in the knowledge graph for a given question."""
    entities: list[str] = Field(description="Entity names to search for in the graph")
    relationship_types: list[str] = Field(
        default=[], description="Relationship types relevant to the question"
    )


class RAGScore(BaseModel):
    score: float = Field(description="Score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation")


class RAGEvaluation(BaseModel):
    faithfulness: RAGScore
    answer_relevancy: RAGScore
    context_precision: RAGScore
    context_recall: RAGScore


# ── Graph RAG system ─────────────────────────────────────────────────────

class GraphRAG:
    """Graph RAG using Neo4j for knowledge graph-based retrieval."""

    def __init__(self, uri: str, user: str, password: str, provider: str = "openai", model: str = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.provider = provider
        self.model = model

    def close(self):
        self.driver.close()

    def build_graph(self, gold_standard: dict):
        """Build the knowledge graph from gold standard documents."""
        print("  Building knowledge graph from documents...")

        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")

            for doc in gold_standard["documents"]:
                print(f"    Processing: {doc['title']}...")

                # Extract entities and relationships using LLM
                extraction = chat_completion_structured(
                    prompt=f"Extract all entities and relationships from this text:\n\n{doc['text']}",
                    output_schema=ExtractionResult,
                    system="Extract entities (PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT) "
                           "and their relationships precisely.",
                    provider=self.provider,
                    model=self.model,
                )

                # Create nodes
                for entity in extraction.entities:
                    session.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.entity_type = $type, e.doc_id = $doc_id
                    """, {"name": entity.name, "type": entity.type, "doc_id": doc["doc_id"]})

                # Create relationships
                for rel in extraction.relationships:
                    session.run("""
                        MATCH (s:Entity {name: $source})
                        MATCH (t:Entity {name: $target})
                        MERGE (s)-[r:RELATES_TO {rel_type: $rel_type}]->(t)
                        SET r.description = $description
                    """, {
                        "source": rel.source,
                        "target": rel.target,
                        "rel_type": rel.type,
                        "description": rel.description,
                    })

                # Also store the full document text for context
                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.text = $text, d.title = $title
                """, {"doc_id": doc["doc_id"], "text": doc["text"], "title": doc["title"]})

            # Count results
            node_count = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            edge_count = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
            print(f"  Graph built: {node_count} entities, {edge_count} relationships")

    def retrieve_context(self, question: str) -> str:
        """Retrieve relevant context from the knowledge graph for a question.

        Strategy:
        1. Use LLM to identify key entities in the question
        2. Look up those entities and their neighborhoods in the graph
        3. Gather connected facts as context
        """
        # Step 1: Extract query entities
        query_analysis = chat_completion_structured(
            prompt=f"What entities should we look up in a knowledge graph to answer this question?\n\n{question}",
            output_schema=QueryEntities,
            system="Identify the key entity names to search for. Use canonical names.",
            provider=self.provider,
            model=self.model,
        )

        context_parts = []

        with self.driver.session() as session:
            for entity_name in query_analysis.entities:
                # Find the entity (fuzzy match)
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                       OR toLower($name) CONTAINS toLower(e.name)
                    RETURN e.name AS name, e.entity_type AS type
                    LIMIT 3
                """, {"name": entity_name})

                matched_entities = list(result)

                for entity in matched_entities:
                    # Get entity's neighborhood (1-hop)
                    result = session.run("""
                        MATCH (e:Entity {name: $name})-[r:RELATES_TO]-(other:Entity)
                        RETURN e.name AS entity, e.entity_type AS entity_type,
                               r.rel_type AS relationship, r.description AS rel_desc,
                               other.name AS other_entity, other.entity_type AS other_type,
                               startNode(r) = e AS is_outgoing
                    """, {"name": entity["name"]})

                    relationships = list(result)
                    if relationships:
                        context_parts.append(f"Entity: {entity['name']} (Type: {entity['type']})")
                        for rel in relationships:
                            if rel["is_outgoing"]:
                                fact = f"  {rel['entity']} --[{rel['relationship']}]--> {rel['other_entity']}"
                            else:
                                fact = f"  {rel['other_entity']} --[{rel['relationship']}]--> {rel['entity']}"
                            if rel["rel_desc"]:
                                fact += f" ({rel['rel_desc']})"
                            context_parts.append(fact)
                        context_parts.append("")

                    # Also get 2-hop connections for multi-hop questions
                    result = session.run("""
                        MATCH (e:Entity {name: $name})-[r1:RELATES_TO]-(mid:Entity)-[r2:RELATES_TO]-(far:Entity)
                        WHERE far <> e
                        RETURN e.name AS start, r1.rel_type AS rel1, mid.name AS mid_entity,
                               r2.rel_type AS rel2, far.name AS end_entity
                        LIMIT 10
                    """, {"name": entity["name"]})

                    hops = list(result)
                    if hops:
                        context_parts.append(f"  Extended connections from {entity['name']}:")
                        for hop in hops:
                            context_parts.append(
                                f"    {hop['start']} -[{hop['rel1']}]-> {hop['mid_entity']} "
                                f"-[{hop['rel2']}]-> {hop['end_entity']}"
                            )
                        context_parts.append("")

            # Also retrieve relevant document text
            for entity_name in query_analysis.entities[:3]:
                result = session.run("""
                    MATCH (e:Entity)-[]-(d:Document)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                    RETURN DISTINCT d.text AS text, d.title AS title
                    LIMIT 2
                """, {"name": entity_name})

                # Fallback: search document text directly
                if not list(result):
                    result = session.run("""
                        MATCH (d:Document)
                        WHERE toLower(d.text) CONTAINS toLower($name)
                        RETURN d.text AS text, d.title AS title
                        LIMIT 2
                    """, {"name": entity_name})

                    for doc in result:
                        context_parts.append(f"Source document ({doc['title']}): {doc['text'][:300]}...")

        return "\n".join(context_parts) if context_parts else "No relevant context found in the knowledge graph."

    def answer(self, question: str) -> dict:
        """Answer a question using Graph RAG."""
        context = self.retrieve_context(question)

        prompt = f"""Answer the following question based on the knowledge graph context provided.
The context includes entity relationships and facts from a knowledge graph.

Knowledge Graph Context:
{context}

Question: {question}

Answer thoroughly based on the graph context:"""

        answer = chat_completion(
            prompt=prompt,
            system="You are a helpful assistant answering questions using knowledge graph context. "
                   "Use the entity relationships and facts provided to construct a comprehensive answer.",
            provider=self.provider,
            model=self.model,
        )

        return {
            "question": question,
            "answer": answer,
            "context": context,
        }


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_answer(question, answer, context, reference_answer, provider, model=None):
    """Evaluate a RAG answer using LLM-as-judge."""
    prompt = f"""Evaluate the quality of this Graph RAG answer.

Question: {question}
Retrieved Context (from knowledge graph): {context[:2000]}
Generated Answer: {answer}
Reference Answer: {reference_answer}

Score each from 0.0 to 1.0:
1. Faithfulness: Is the answer grounded in the retrieved context?
2. Answer Relevancy: Does it address the question?
3. Context Precision: Is the retrieved context relevant?
4. Context Recall: Does the context contain needed information?"""

    return chat_completion_structured(
        prompt=prompt,
        output_schema=RAGEvaluation,
        system="You are an expert RAG evaluator. Score precisely.",
        provider=provider,
        model=model,
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph RAG Evaluation")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--build-graph", action="store_true", help="Build the graph from scratch")
    parser.add_argument("--uri", type=str, default=NEO4J_URI)
    parser.add_argument("--user", type=str, default=NEO4J_USER)
    parser.add_argument("--password", type=str, default=NEO4J_PASSWORD)
    args = parser.parse_args()

    print("=" * 70)
    print("  KG Evaluation: Graph RAG (Neo4j)")
    print("=" * 70)
    print(f"\n  Provider: {args.provider}\n")

    # Load data
    with open(GOLD_STANDARD_PATH) as f:
        gold_standard = json.load(f)
    with open(EVAL_QUESTIONS_PATH) as f:
        eval_data = json.load(f)

    questions = eval_data["questions"]

    # Build Graph RAG
    graph_rag = GraphRAG(args.uri, args.user, args.password, provider=args.provider, model=args.model)

    try:
        graph_rag.driver.verify_connectivity()
    except Exception as e:
        print(f"  Error: Cannot connect to Neo4j: {e}")
        print("  Make sure Neo4j is running: docker compose up -d")
        sys.exit(1)

    if args.build_graph:
        graph_rag.build_graph(gold_standard)

    # Run evaluation
    results = []
    type_scores = {}

    for i, q in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {q['question'][:60]}...")

        rag_result = graph_rag.answer(q["question"])

        evaluation = evaluate_answer(
            question=q["question"],
            answer=rag_result["answer"],
            context=rag_result["context"],
            reference_answer=q["reference_answer"],
            provider=args.provider,
            model=args.model,
        )

        scores = {
            "faithfulness": evaluation.faithfulness.score,
            "answer_relevancy": evaluation.answer_relevancy.score,
            "context_precision": evaluation.context_precision.score,
            "context_recall": evaluation.context_recall.score,
        }

        print(f"    Faith: {scores['faithfulness']:.2f}  "
              f"Relev: {scores['answer_relevancy']:.2f}  "
              f"Prec: {scores['context_precision']:.2f}  "
              f"Recall: {scores['context_recall']:.2f}")

        result_entry = {
            "question_id": q["id"],
            "question_type": q["type"],
            "question": q["question"],
            "answer": rag_result["answer"],
            "reference_answer": q["reference_answer"],
            "scores": scores,
            "evaluation_details": {
                "faithfulness_reasoning": evaluation.faithfulness.reasoning,
                "relevancy_reasoning": evaluation.answer_relevancy.reasoning,
                "precision_reasoning": evaluation.context_precision.reasoning,
                "recall_reasoning": evaluation.context_recall.reasoning,
            },
        }
        results.append(result_entry)

        q_type = q["type"]
        if q_type not in type_scores:
            type_scores[q_type] = []
        type_scores[q_type].append(scores)

    # Print summary
    print(f"\n\n{'='*70}")
    print("  GRAPH RAG EVALUATION SUMMARY")
    print(f"{'='*70}")

    all_scores = [r["scores"] for r in results]
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print(f"\n  Overall Averages:")
    overall_avg = {}
    for m in metrics:
        avg = sum(s[m] for s in all_scores) / len(all_scores)
        overall_avg[m] = round(avg, 4)
        print(f"    {m:<22} {avg:.4f}")

    print(f"\n  Per Question Type:")
    type_averages = {}
    for q_type in sorted(type_scores.keys()):
        scores_list = type_scores[q_type]
        type_avg = {}
        print(f"\n    {q_type} ({len(scores_list)} questions):")
        for m in metrics:
            avg = sum(s[m] for s in scores_list) / len(scores_list)
            type_avg[m] = round(avg, 4)
            print(f"      {m:<22} {avg:.4f}")
        type_averages[q_type] = type_avg

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "system": "graph_rag",
        "provider": args.provider,
        "overall_averages": overall_avg,
        "per_type_averages": type_averages,
        "per_question_results": results,
    }
    output_path = OUTPUT_DIR / "graph_rag_eval.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    graph_rag.close()


if __name__ == "__main__":
    main()
