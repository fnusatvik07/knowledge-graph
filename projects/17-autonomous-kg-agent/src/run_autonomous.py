"""Entry point for the Autonomous KG Agent.

Initializes with a seed document, runs the autonomous agent for N iterations,
and displays a growth trajectory of the KG over time.

Usage:
    python src/run_autonomous.py
    python src/run_autonomous.py --iterations 10
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from shared.llm_clients import chat_completion_structured
from shared.config import get_neo4j_config

from src.agents.autonomous_agent import run_autonomous, ExtractionResult, _write_extraction_to_neo4j, _extract_from_text
from src.tools.kg_analyzer import get_kg_stats, get_quality_score, get_coverage_score
from src.tools.source_manager import reset_processing_state


# ── Configuration ────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


def clear_kg():
    """Clear all entities and relationships from the KG."""
    config = get_neo4j_config()
    driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    print("Cleared existing KG data.")


def ingest_seed_document():
    """Ingest the seed document into the KG."""
    seed_path = DATA_DIR / "seed_topic.txt"
    with open(seed_path) as f:
        seed_text = f.read()

    print(f"\nIngesting seed document ({len(seed_text.split())} words)...")
    extraction = _extract_from_text(seed_text, "The History of Artificial Intelligence")
    result = _write_extraction_to_neo4j(extraction, "seed_document")
    print(f"  Extracted {result['entities']} entities, {result['relationships']} relationships")


def print_stats(label: str = ""):
    """Print current KG statistics."""
    stats = get_kg_stats()
    quality = get_quality_score()
    coverage = get_coverage_score()

    print(f"\n--- KG Stats{' (' + label + ')' if label else ''} ---")
    print(f"  Entities: {stats['total_entities']}")
    print(f"  Relationships: {stats['total_relationships']}")
    print(f"  Entity types: {stats.get('entity_types', {})}")
    print(f"  Avg degree: {stats.get('avg_degree', 0):.2f}")
    print(f"  Quality: {quality:.3f}")
    print(f"  Coverage: {coverage:.3f}")

    return {
        "entities": stats["total_entities"],
        "relationships": stats["total_relationships"],
        "quality": quality,
        "coverage": coverage,
    }


def plot_growth_trajectory(trajectory: list[dict]):
    """Plot KG growth over iterations using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed. Skipping growth plot.")
        print("Install with: pip install matplotlib")
        return

    iterations = list(range(len(trajectory)))
    entities = [t["entities"] for t in trajectory]
    relationships = [t["relationships"] for t in trajectory]
    quality = [t["quality"] for t in trajectory]
    coverage = [t["coverage"] for t in trajectory]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Autonomous KG Agent -- Growth Trajectory", fontsize=14, fontweight="bold")

    # Entities over time
    axes[0, 0].plot(iterations, entities, "b-o", linewidth=2, markersize=6)
    axes[0, 0].set_title("Entities")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

    # Relationships over time
    axes[0, 1].plot(iterations, relationships, "g-o", linewidth=2, markersize=6)
    axes[0, 1].set_title("Relationships")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(True, alpha=0.3)

    # Quality score over time
    axes[1, 0].plot(iterations, quality, "r-o", linewidth=2, markersize=6)
    axes[1, 0].set_title("Quality Score")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Score (0-1)")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)

    # Coverage score over time
    axes[1, 1].plot(iterations, coverage, "m-o", linewidth=2, markersize=6)
    axes[1, 1].set_title("Coverage Score")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Score (0-1)")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "growth_trajectory.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nGrowth trajectory plot saved to: {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Autonomous KG Agent")
    parser.add_argument("--iterations", type=int, default=5, help="Number of autonomous iterations")
    parser.add_argument("--threshold", type=float, default=0.8, help="Quality threshold to stop")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing KG data")
    args = parser.parse_args()

    print("=" * 70)
    print("  AUTONOMOUS KG AGENT")
    print("=" * 70)
    print(f"  Max iterations: {args.iterations}")
    print(f"  Quality threshold: {args.threshold}")

    # ── Setup ────────────────────────────────────────────────────────
    if not args.no_clear:
        clear_kg()
        reset_processing_state()

    # ── Step 1: Ingest seed document ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: Seed Ingestion")
    print("=" * 70)

    ingest_seed_document()
    trajectory = [print_stats("after seed")]

    # ── Step 2: Run autonomous agent ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: Autonomous Improvement Loop")
    print("=" * 70)

    final_state = run_autonomous(
        max_iterations=args.iterations,
        quality_threshold=args.threshold,
    )

    # ── Display results ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ACTION HISTORY")
    print("=" * 70)

    for action in final_state["action_history"]:
        iteration = action.get("iteration", "?")
        action_type = action.get("action_type", "unknown")
        result = action.get("result", "")[:120]
        q_before = action.get("quality_before", "?")
        q_after = action.get("quality_after", "?")

        print(f"\n  Iteration {iteration}: {action_type}")
        print(f"    Result: {result}")
        if isinstance(q_before, float) and isinstance(q_after, float):
            delta = q_after - q_before
            arrow = "+" if delta >= 0 else ""
            print(f"    Quality: {q_before:.3f} -> {q_after:.3f} ({arrow}{delta:.3f})")

        # Record stats for trajectory
        trajectory.append({
            "entities": final_state["kg_stats"].get("total_entities", 0),
            "relationships": final_state["kg_stats"].get("total_relationships", 0),
            "quality": action.get("quality_after", 0),
            "coverage": final_state["kg_stats"].get("coverage_score", 0),
        })

    # ── Final stats ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL STATE")
    print("=" * 70)

    final_stats = print_stats("final")
    trajectory.append(final_stats)

    print(f"\n  Total iterations: {final_state['iteration']}")
    print(f"  Total actions: {len(final_state['action_history'])}")
    print(f"  Final quality: {final_state['quality_score']:.3f}")

    # ── Save trajectory data ─────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectory_path = OUTPUT_DIR / "trajectory.json"
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2, default=str)
    print(f"\n  Trajectory data saved to: {trajectory_path}")

    # ── Plot growth ──────────────────────────────────────────────────
    plot_growth_trajectory(trajectory)

    print("\n" + "=" * 70)
    print("  Demo complete. Explore the KG at http://localhost:7474")
    print("=" * 70)


if __name__ == "__main__":
    main()
