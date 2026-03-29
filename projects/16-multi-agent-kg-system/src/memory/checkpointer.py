"""LangGraph checkpointer backed by Neo4j.

Implements a checkpoint saver that persists LangGraph state to Neo4j,
enabling interrupted workflows to be resumed.

Reference: https://langchain-ai.github.io/langgraph/concepts/persistence/
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional, Iterator

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langchain_core.runnables import RunnableConfig
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.config import get_neo4j_config


class Neo4jCheckpointer(BaseCheckpointSaver):
    """A LangGraph checkpoint saver that stores graph state in Neo4j.

    Checkpoints are stored as nodes with relationships to their parent
    checkpoints, forming a checkpoint history tree.
    """

    def __init__(self):
        super().__init__()
        config = get_neo4j_config()
        self._driver = GraphDatabase.driver(
            config["uri"], auth=(config["user"], config["password"])
        )
        self._ensure_schema()

    def _ensure_schema(self):
        """Create indexes for checkpoint nodes."""
        with self._driver.session() as session:
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (c:Checkpoint) ON (c.thread_id, c.checkpoint_id)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (c:Checkpoint) ON (c.thread_id, c.checkpoint_ns)"
            )

    def _serialize(self, obj: Any) -> str:
        """Serialize an object to JSON string."""
        if obj is None:
            return "null"
        try:
            return json.dumps(obj, default=str)
        except (TypeError, ValueError):
            return json.dumps(str(obj))

    def _deserialize(self, data: str) -> Any:
        """Deserialize a JSON string back to an object."""
        if data is None or data == "null":
            return None
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict] = None,
    ) -> RunnableConfig:
        """Save a checkpoint to Neo4j.

        Args:
            config: The runnable config containing thread_id
            checkpoint: The checkpoint data to save
            metadata: Metadata about the checkpoint
            new_versions: Channel version updates

        Returns:
            Updated config with the checkpoint_id
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        now = datetime.now(timezone.utc).isoformat()

        with self._driver.session() as session:
            session.run(
                """
                MERGE (c:Checkpoint {thread_id: $thread_id, checkpoint_id: $checkpoint_id})
                SET c.checkpoint_ns = $checkpoint_ns,
                    c.checkpoint_data = $checkpoint_data,
                    c.metadata = $metadata,
                    c.channel_versions = $channel_versions,
                    c.created_at = $now
                """,
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_data=self._serialize(checkpoint),
                metadata=self._serialize(metadata),
                channel_versions=self._serialize(new_versions),
                now=now,
            )

            # Link to parent checkpoint if exists
            if parent_checkpoint_id:
                session.run(
                    """
                    MATCH (child:Checkpoint {thread_id: $thread_id, checkpoint_id: $child_id})
                    MATCH (parent:Checkpoint {thread_id: $thread_id, checkpoint_id: $parent_id})
                    MERGE (child)-[:PARENT_CHECKPOINT]->(parent)
                    """,
                    thread_id=thread_id,
                    child_id=checkpoint_id,
                    parent_id=parent_checkpoint_id,
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple by config.

        Args:
            config: Config with thread_id and optional checkpoint_id

        Returns:
            CheckpointTuple or None if not found
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        with self._driver.session() as session:
            if checkpoint_id:
                result = session.run(
                    """
                    MATCH (c:Checkpoint {thread_id: $thread_id, checkpoint_id: $checkpoint_id})
                    OPTIONAL MATCH (c)-[:PARENT_CHECKPOINT]->(p:Checkpoint)
                    RETURN c.checkpoint_data AS checkpoint_data,
                           c.metadata AS metadata,
                           c.checkpoint_id AS checkpoint_id,
                           c.checkpoint_ns AS checkpoint_ns,
                           p.checkpoint_id AS parent_checkpoint_id
                    """,
                    thread_id=thread_id,
                    checkpoint_id=checkpoint_id,
                )
            else:
                # Get the latest checkpoint for this thread
                result = session.run(
                    """
                    MATCH (c:Checkpoint {thread_id: $thread_id})
                    WHERE c.checkpoint_ns = $checkpoint_ns
                    OPTIONAL MATCH (c)-[:PARENT_CHECKPOINT]->(p:Checkpoint)
                    RETURN c.checkpoint_data AS checkpoint_data,
                           c.metadata AS metadata,
                           c.checkpoint_id AS checkpoint_id,
                           c.checkpoint_ns AS checkpoint_ns,
                           p.checkpoint_id AS parent_checkpoint_id
                    ORDER BY c.created_at DESC
                    LIMIT 1
                    """,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                )

            record = result.single()
            if record is None:
                return None

            checkpoint_data = self._deserialize(record["checkpoint_data"])
            metadata = self._deserialize(record["metadata"])
            parent_id = record["parent_checkpoint_id"]

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": record["checkpoint_ns"] or "",
                        "checkpoint_id": record["checkpoint_id"],
                    }
                },
                checkpoint=checkpoint_data,
                metadata=metadata,
                parent_config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": record["checkpoint_ns"] or "",
                        "checkpoint_id": parent_id,
                    }
                } if parent_id else None,
            )

    def list(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints, optionally filtered.

        Args:
            config: Filter by thread_id
            filter: Additional metadata filters
            before: List checkpoints before this one
            limit: Maximum number to return

        Yields:
            CheckpointTuple instances
        """
        query_parts = ["MATCH (c:Checkpoint)"]
        params = {}

        if config:
            thread_id = config["configurable"]["thread_id"]
            query_parts.append("WHERE c.thread_id = $thread_id")
            params["thread_id"] = thread_id

        query_parts.append("OPTIONAL MATCH (c)-[:PARENT_CHECKPOINT]->(p:Checkpoint)")
        query_parts.append(
            "RETURN c.checkpoint_data AS checkpoint_data, "
            "c.metadata AS metadata, "
            "c.checkpoint_id AS checkpoint_id, "
            "c.checkpoint_ns AS checkpoint_ns, "
            "c.thread_id AS thread_id, "
            "p.checkpoint_id AS parent_checkpoint_id "
            "ORDER BY c.created_at DESC"
        )

        if limit:
            query_parts.append(f"LIMIT {limit}")

        query = "\n".join(query_parts)

        with self._driver.session() as session:
            result = session.run(query, **params)
            for record in result:
                checkpoint_data = self._deserialize(record["checkpoint_data"])
                metadata = self._deserialize(record["metadata"])
                parent_id = record["parent_checkpoint_id"]

                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": record["thread_id"],
                            "checkpoint_ns": record["checkpoint_ns"] or "",
                            "checkpoint_id": record["checkpoint_id"],
                        }
                    },
                    checkpoint=checkpoint_data,
                    metadata=metadata,
                    parent_config={
                        "configurable": {
                            "thread_id": record["thread_id"],
                            "checkpoint_ns": record["checkpoint_ns"] or "",
                            "checkpoint_id": parent_id,
                        }
                    } if parent_id else None,
                )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes for a checkpoint.

        Args:
            config: Config identifying the checkpoint
            writes: List of (channel, value) tuples
            task_id: ID of the task that produced the writes
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._driver.session() as session:
            session.run(
                """
                MATCH (c:Checkpoint {thread_id: $thread_id, checkpoint_id: $checkpoint_id})
                SET c.pending_writes = $writes,
                    c.pending_task_id = $task_id
                """,
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                writes=self._serialize(writes),
                task_id=task_id,
            )

    def close(self):
        """Close the Neo4j driver."""
        self._driver.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
