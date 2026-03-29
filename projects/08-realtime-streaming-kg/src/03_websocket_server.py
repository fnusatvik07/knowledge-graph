"""WebSocket server that broadcasts graph change events to connected clients.

Monitors the change log file written by the stream processor and pushes
notifications to all connected WebSocket clients in real-time. Clients can
subscribe to specific entity types to filter the stream.

websockets library docs: https://websockets.readthedocs.io/

Usage:
    python src/03_websocket_server.py
    python src/03_websocket_server.py --host 0.0.0.0 --port 8765
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import websockets

# ── Configuration ────────────────────────────────────────────────────────
CHANGE_LOG_PATH = Path(__file__).resolve().parent.parent / "output" / "change_log.jsonl"
POLL_INTERVAL = 0.5  # seconds between log file checks


class GraphChangeServer:
    """WebSocket server that broadcasts knowledge graph changes."""

    def __init__(self):
        self.clients: dict[websockets.WebSocketServerProtocol, set[str]] = {}
        self.last_position = 0  # Track file read position

    async def register(self, websocket):
        """Register a new client connection."""
        self.clients[websocket] = set()  # Empty set = subscribe to all
        client_id = id(websocket)
        print(f"  Client {client_id} connected. Total clients: {len(self.clients)}")

        # Send welcome message
        await websocket.send(json.dumps({
            "type": "welcome",
            "message": "Connected to Streaming KG WebSocket server",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

    async def unregister(self, websocket):
        """Unregister a disconnected client."""
        self.clients.pop(websocket, None)
        client_id = id(websocket)
        print(f"  Client {client_id} disconnected. Total clients: {len(self.clients)}")

    async def handle_client(self, websocket):
        """Handle messages from a connected client."""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON",
                    }))
        except websockets.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def handle_message(self, websocket, data: dict):
        """Handle incoming messages from clients.

        Supported message types:
        - subscribe: {"type": "subscribe", "entity_types": ["PERSON", "ORGANIZATION"]}
        - unsubscribe: {"type": "unsubscribe"}
        - get_stats: {"type": "get_stats"}
        """
        msg_type = data.get("type", "")

        if msg_type == "subscribe":
            entity_types = set(data.get("entity_types", []))
            self.clients[websocket] = entity_types
            await websocket.send(json.dumps({
                "type": "subscribed",
                "entity_types": list(entity_types) if entity_types else ["ALL"],
            }))

        elif msg_type == "unsubscribe":
            self.clients[websocket] = set()
            await websocket.send(json.dumps({
                "type": "unsubscribed",
                "message": "Subscribed to all events",
            }))

        elif msg_type == "get_stats":
            stats = self.get_change_stats()
            await websocket.send(json.dumps({
                "type": "stats",
                "data": stats,
            }))

    def get_change_stats(self) -> dict:
        """Get statistics from the change log."""
        if not CHANGE_LOG_PATH.exists():
            return {"total_changes": 0, "entities_added": 0, "relationships_added": 0, "facts_invalidated": 0}

        stats = {"total_changes": 0, "entities_added": 0, "relationships_added": 0, "facts_invalidated": 0}
        with open(CHANGE_LOG_PATH, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    stats["total_changes"] += 1
                    if entry["type"] == "entity_added":
                        stats["entities_added"] += 1
                    elif entry["type"] == "relationship_added":
                        stats["relationships_added"] += 1
                    elif entry["type"] == "fact_invalidated":
                        stats["facts_invalidated"] += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        return stats

    def should_send_to_client(self, client_filters: set[str], change_data: dict) -> bool:
        """Check if a change event matches a client's subscription filters."""
        if not client_filters:
            return True  # Empty filter = receive all

        # Check if entity type matches
        entity_type = change_data.get("data", {}).get("type", "")
        return entity_type in client_filters

    async def broadcast(self, change_entry: dict):
        """Broadcast a change event to all subscribed clients."""
        if not self.clients:
            return

        message = json.dumps(change_entry)
        disconnected = []

        for websocket, filters in self.clients.items():
            if self.should_send_to_client(filters, change_entry):
                try:
                    await websocket.send(message)
                except websockets.ConnectionClosed:
                    disconnected.append(websocket)

        for ws in disconnected:
            await self.unregister(ws)

    async def watch_change_log(self):
        """Continuously monitor the change log file for new entries."""
        print(f"  Monitoring change log: {CHANGE_LOG_PATH}")

        while True:
            try:
                if CHANGE_LOG_PATH.exists():
                    with open(CHANGE_LOG_PATH, "r") as f:
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        self.last_position = f.tell()

                    for line in new_lines:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                await self.broadcast(entry)
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"  Error reading change log: {e}")

            await asyncio.sleep(POLL_INTERVAL)


async def run_server(host: str = "localhost", port: int = 8765):
    """Start the WebSocket server and change log monitor."""
    server = GraphChangeServer()

    print("=" * 60)
    print("Streaming KG — WebSocket Server")
    print("=" * 60)
    print(f"\n  Server: ws://{host}:{port}")
    print(f"  Change log: {CHANGE_LOG_PATH}")
    print("\n  Waiting for connections...\n")

    # Start the change log watcher as a background task
    asyncio.create_task(server.watch_change_log())

    # Start the WebSocket server
    async with websockets.serve(server.handle_client, host, port):
        await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(description="WebSocket server for Streaming KG")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    args = parser.parse_args()

    try:
        asyncio.run(run_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
