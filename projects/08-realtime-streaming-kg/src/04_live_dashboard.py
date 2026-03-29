"""Terminal-based live dashboard for the Real-Time Streaming Knowledge Graph.

Connects to the WebSocket server and displays real-time statistics:
- Total nodes and edges in the graph
- Recent entity/relationship additions
- Entity type distribution
- Invalidated facts

Updates live as new data streams in from the processor.

websockets docs: https://websockets.readthedocs.io/

Usage:
    python src/04_live_dashboard.py
    python src/04_live_dashboard.py --ws-url ws://localhost:8765
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import websockets

# ── Dashboard state ──────────────────────────────────────────────────────

class DashboardState:
    """Tracks the current state of the knowledge graph for display."""

    def __init__(self):
        self.total_entities = 0
        self.total_relationships = 0
        self.total_invalidated = 0
        self.entity_type_counts: dict[str, int] = defaultdict(int)
        self.recent_events: list[dict] = []
        self.max_recent = 15
        self.last_update = None
        self.connected = False

    def add_event(self, event: dict):
        """Process a new event from the WebSocket server."""
        self.last_update = datetime.now().strftime("%H:%M:%S")
        event_type = event.get("type", "")

        if event_type == "entity_added":
            self.total_entities += 1
            entity_type = event.get("data", {}).get("type", "UNKNOWN")
            self.entity_type_counts[entity_type] += 1
            self.recent_events.append({
                "time": self.last_update,
                "desc": f"+ Entity: {event['data'].get('name', '?')} ({entity_type})",
            })

        elif event_type == "relationship_added":
            self.total_relationships += 1
            data = event.get("data", {})
            self.recent_events.append({
                "time": self.last_update,
                "desc": f"+ Rel: {data.get('source', '?')} --[{data.get('type', '?')}]--> {data.get('target', '?')}",
            })

        elif event_type == "fact_invalidated":
            self.total_invalidated += 1
            data = event.get("data", {})
            self.recent_events.append({
                "time": self.last_update,
                "desc": f"X Invalidated: {data.get('reason', '?')[:60]}",
            })

        # Keep only recent events
        if len(self.recent_events) > self.max_recent:
            self.recent_events = self.recent_events[-self.max_recent:]

    def render(self) -> str:
        """Render the dashboard as a string for terminal display."""
        lines = []
        lines.append("\033[2J\033[H")  # Clear screen, move cursor to top
        lines.append("=" * 70)
        lines.append("  STREAMING KNOWLEDGE GRAPH — LIVE DASHBOARD")
        lines.append("=" * 70)
        lines.append("")

        # Connection status
        status = "CONNECTED" if self.connected else "DISCONNECTED"
        status_color = "\033[92m" if self.connected else "\033[91m"
        lines.append(f"  Status: {status_color}{status}\033[0m")
        if self.last_update:
            lines.append(f"  Last Update: {self.last_update}")
        lines.append("")

        # Summary stats
        lines.append("-" * 70)
        lines.append("  GRAPH STATISTICS")
        lines.append("-" * 70)
        lines.append(f"  Total Entities:      {self.total_entities}")
        lines.append(f"  Total Relationships: {self.total_relationships}")
        lines.append(f"  Invalidated Facts:   {self.total_invalidated}")
        lines.append("")

        # Entity type distribution
        if self.entity_type_counts:
            lines.append("-" * 70)
            lines.append("  ENTITY TYPES")
            lines.append("-" * 70)
            for etype, count in sorted(self.entity_type_counts.items(), key=lambda x: -x[1]):
                bar = "#" * min(count * 3, 40)
                lines.append(f"  {etype:<20} {count:>4}  {bar}")
            lines.append("")

        # Recent events
        lines.append("-" * 70)
        lines.append("  RECENT EVENTS")
        lines.append("-" * 70)
        if self.recent_events:
            for event in reversed(self.recent_events[-10:]):
                lines.append(f"  [{event['time']}] {event['desc'][:55]}")
        else:
            lines.append("  Waiting for events...")
        lines.append("")

        lines.append("-" * 70)
        lines.append("  Press Ctrl+C to exit")
        lines.append("-" * 70)

        return "\n".join(lines)


# ── WebSocket client ─────────────────────────────────────────────────────

async def run_dashboard(ws_url: str):
    """Connect to WebSocket server and run the live dashboard."""
    state = DashboardState()

    print("Connecting to WebSocket server...")
    print(f"URL: {ws_url}\n")

    retry_delay = 2

    while True:
        try:
            async with websockets.connect(ws_url) as websocket:
                state.connected = True
                print(state.render())

                # Request initial stats
                await websocket.send(json.dumps({"type": "get_stats"}))

                async for message in websocket:
                    try:
                        data = json.loads(message)

                        if data.get("type") == "welcome":
                            continue
                        elif data.get("type") == "stats":
                            # Update from stats response
                            stats = data.get("data", {})
                            state.total_entities = stats.get("entities_added", state.total_entities)
                            state.total_relationships = stats.get("relationships_added", state.total_relationships)
                            state.total_invalidated = stats.get("facts_invalidated", state.total_invalidated)
                        else:
                            state.add_event(data)

                        # Re-render the dashboard
                        print(state.render())

                    except json.JSONDecodeError:
                        continue

        except websockets.ConnectionClosed:
            state.connected = False
            print(state.render())
            print(f"\n  Connection lost. Reconnecting in {retry_delay}s...")
            await asyncio.sleep(retry_delay)

        except ConnectionRefusedError:
            state.connected = False
            print(f"Cannot connect to {ws_url}. Retrying in {retry_delay}s...")
            print("Make sure the WebSocket server is running: python src/03_websocket_server.py")
            await asyncio.sleep(retry_delay)

        except KeyboardInterrupt:
            break


def main():
    parser = argparse.ArgumentParser(description="Live Dashboard for Streaming KG")
    parser.add_argument("--ws-url", type=str, default="ws://localhost:8765", help="WebSocket server URL")
    args = parser.parse_args()

    try:
        asyncio.run(run_dashboard(args.ws_url))
    except KeyboardInterrupt:
        print("\nDashboard closed.")


if __name__ == "__main__":
    main()
