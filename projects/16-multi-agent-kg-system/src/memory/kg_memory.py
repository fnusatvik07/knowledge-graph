"""KG-backed agent memory -- persists agent state, session context, and
learning history to Neo4j.

Supports:
- Saving/restoring agent state across runs
- Memory summarization when entries grow large
- Cross-session learning (tracking prompt effectiveness, source reliability)
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import chat_completion
from shared.config import get_neo4j_config


class KGMemory:
    """Knowledge-graph-backed memory for agents.

    Stores memory entries as nodes in Neo4j with relationships to sessions
    and agents, enabling persistent cross-session learning.
    """

    def __init__(self, agent_name: str, session_id: str | None = None):
        """Initialize KG memory for an agent.

        Args:
            agent_name: Name of the agent (e.g., "extractor", "researcher")
            session_id: Optional session ID. Auto-generated if not provided.
        """
        self.agent_name = agent_name
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        config = get_neo4j_config()
        self._driver = GraphDatabase.driver(
            config["uri"], auth=(config["user"], config["password"])
        )
        self._ensure_schema()

    def _ensure_schema(self):
        """Create indexes and constraints for memory nodes."""
        with self._driver.session() as session:
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (m:MemoryEntry) ON (m.agent_name, m.session_id)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (s:Session) ON (s.session_id)"
            )

    def save_entry(self, key: str, value: Any, entry_type: str = "state") -> None:
        """Save a memory entry to Neo4j.

        Args:
            key: Entry key (e.g., "documents_processed", "last_question")
            value: Entry value (will be JSON-serialized if not a string)
            entry_type: One of "state", "observation", "learning", "metric"
        """
        now = datetime.now(timezone.utc).isoformat()
        value_str = json.dumps(value) if not isinstance(value, str) else value

        with self._driver.session() as session:
            # Ensure session node exists
            session.run(
                """
                MERGE (s:Session {session_id: $session_id})
                SET s.agent_name = $agent_name,
                    s.updated_at = $now
                ON CREATE SET s.created_at = $now
                """,
                session_id=self.session_id,
                agent_name=self.agent_name,
                now=now,
            )
            # Create or update memory entry
            session.run(
                """
                MERGE (m:MemoryEntry {agent_name: $agent_name, session_id: $session_id, key: $key})
                SET m.value = $value,
                    m.entry_type = $entry_type,
                    m.updated_at = $now
                ON CREATE SET m.created_at = $now
                WITH m
                MATCH (s:Session {session_id: $session_id})
                MERGE (s)-[:HAS_MEMORY]->(m)
                """,
                agent_name=self.agent_name,
                session_id=self.session_id,
                key=key,
                value=value_str,
                entry_type=entry_type,
                now=now,
            )

    def get_entry(self, key: str) -> Any | None:
        """Retrieve a memory entry by key.

        Returns:
            The stored value (JSON-deserialized if possible), or None if not found.
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (m:MemoryEntry {agent_name: $agent_name, session_id: $session_id, key: $key})
                RETURN m.value AS value
                """,
                agent_name=self.agent_name,
                session_id=self.session_id,
                key=key,
            )
            record = result.single()
            if record is None:
                return None
            try:
                return json.loads(record["value"])
            except (json.JSONDecodeError, TypeError):
                return record["value"]

    def get_all_entries(self, entry_type: str | None = None) -> list[dict]:
        """Get all memory entries for this agent/session.

        Args:
            entry_type: Optional filter by type

        Returns:
            List of dicts with key, value, entry_type, updated_at
        """
        query = """
            MATCH (m:MemoryEntry {agent_name: $agent_name, session_id: $session_id})
        """
        params = {"agent_name": self.agent_name, "session_id": self.session_id}

        if entry_type:
            query += " WHERE m.entry_type = $entry_type"
            params["entry_type"] = entry_type

        query += " RETURN m.key AS key, m.value AS value, m.entry_type AS entry_type, m.updated_at AS updated_at"

        with self._driver.session() as session:
            result = session.run(query, **params)
            entries = []
            for record in result:
                try:
                    value = json.loads(record["value"])
                except (json.JSONDecodeError, TypeError):
                    value = record["value"]
                entries.append({
                    "key": record["key"],
                    "value": value,
                    "entry_type": record["entry_type"],
                    "updated_at": record["updated_at"],
                })
            return entries

    def save_session_context(self, context: dict) -> None:
        """Save the full session context for later restoration.

        Args:
            context: Dictionary of agent state to persist
        """
        self.save_entry("_session_context", context, entry_type="state")

    def restore_session_context(self) -> dict | None:
        """Restore the session context from a previous run.

        Returns:
            The saved context dict, or None if no previous session found.
        """
        return self.get_entry("_session_context")

    def summarize_old_entries(self, max_entries: int = 50) -> str | None:
        """When memory grows large, summarize old entries using LLM.

        Args:
            max_entries: Threshold above which summarization is triggered

        Returns:
            The summary string, or None if summarization was not needed.
        """
        entries = self.get_all_entries()
        if len(entries) <= max_entries:
            return None

        # Take the oldest entries to summarize
        entries_to_summarize = entries[:len(entries) - max_entries // 2]
        entries_text = "\n".join(
            f"- [{e['entry_type']}] {e['key']}: {e['value']}"
            for e in entries_to_summarize
        )

        summary = chat_completion(
            prompt=f"""Summarize the following agent memory entries into a concise summary
that preserves the key learnings and important state:

Agent: {self.agent_name}
Session: {self.session_id}

Entries:
{entries_text}""",
            system="You are summarizing agent memory. Preserve critical facts and learnings.",
        )

        # Save summary and delete old entries
        self.save_entry("_memory_summary", summary, entry_type="state")

        with self._driver.session() as session:
            for entry in entries_to_summarize:
                session.run(
                    """
                    MATCH (m:MemoryEntry {
                        agent_name: $agent_name,
                        session_id: $session_id,
                        key: $key
                    })
                    WHERE m.key <> '_memory_summary' AND m.key <> '_session_context'
                    DETACH DELETE m
                    """,
                    agent_name=self.agent_name,
                    session_id=self.session_id,
                    key=entry["key"],
                )

        return summary

    def record_learning(self, key: str, observation: str, metric: float | None = None) -> None:
        """Record a learning observation for cross-session improvement.

        Args:
            key: What was learned about (e.g., "extraction_prompt_v2")
            observation: What was observed
            metric: Optional numeric metric (e.g., accuracy score)
        """
        value = {"observation": observation}
        if metric is not None:
            value["metric"] = metric
        self.save_entry(f"learning_{key}", value, entry_type="learning")

    def get_learnings(self) -> list[dict]:
        """Get all cross-session learnings.

        Returns:
            List of learning entries across all sessions for this agent.
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (m:MemoryEntry {agent_name: $agent_name})
                WHERE m.entry_type = 'learning'
                RETURN m.key AS key, m.value AS value,
                       m.session_id AS session_id, m.updated_at AS updated_at
                ORDER BY m.updated_at DESC
                """,
                agent_name=self.agent_name,
            )
            learnings = []
            for record in result:
                try:
                    value = json.loads(record["value"])
                except (json.JSONDecodeError, TypeError):
                    value = record["value"]
                learnings.append({
                    "key": record["key"],
                    "value": value,
                    "session_id": record["session_id"],
                    "updated_at": record["updated_at"],
                })
            return learnings

    def close(self):
        """Close the Neo4j driver connection."""
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
