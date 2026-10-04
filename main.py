#!/usr/bin/env python3
"""
IFGamePlayer - Graph-Augmented Agent for Interactive Fiction (ADVENT / Colossal Cave)
Runs natively using Python Standard Library (compatible with NetBSD / BSD / Linux on sdf.org).
"""

import hashlib
import json
import os
import re
import random
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, Any

# =====================================================================
# 1. GRAPH & STATE REPRESENTATION
# =====================================================================

class RoomNode:
    def __init__(self, room_id: str, description: str):
        self.room_id = room_id
        self.description = description
        self.items: Set[str] = set()
        self.exits: Dict[str, str] = {}  # {action: target_room_id}
        self.lifetime_visits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "description": self.description,
            "items": sorted(list(self.items)),
            "exits": self.exits,
            "lifetime_visits": self.lifetime_visits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoomNode":
        node = cls(room_id=data["room_id"], description=data.get("description", ""))
        node.items = set(data.get("items", []))
        node.exits = data.get("exits", {})
        node.lifetime_visits = data.get("lifetime_visits", 0)
        return node


class WorldGraph:
    def __init__(self):
        self.nodes: Dict[str, RoomNode] = {}
        self.current_room_id: Optional[str] = None
        self.inventory: Set[str] = set()

    def get_or_create_node(self, room_id: str, description: str) -> RoomNode:
        if room_id not in self.nodes:
            self.nodes[room_id] = RoomNode(room_id, description)
        return self.nodes[room_id]

    def add_transition(self, from_id: str, action: str, to_id: str):
        if from_id in self.nodes:
            self.nodes[from_id].exits[action] = to_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_room_id": self.current_room_id,
            "inventory": sorted(list(self.inventory)),
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldGraph":
        graph = cls()
        graph.current_room_id = data.get("current_room_id")
        graph.inventory = set(data.get("inventory", []))
        nodes_data = data.get("nodes", {})
        for nid, ndict in nodes_data.items():
            graph.nodes[nid] = RoomNode.from_dict(ndict)
        return graph


# =====================================================================
# 2. DISK SERIALIZATION & STORAGE
# =====================================================================

class GraphStorage:
    @staticmethod
    def save_graph(graph: WorldGraph, filepath: str) -> None:
        temp_path = f"{filepath}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(graph.to_dict(), f, indent=2)
            os.replace(temp_path, filepath)
            print(f"[Storage] Graph saved successfully ({len(graph.nodes)} rooms mapped).")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"[Storage] Save failed: {e}")

    @staticmethod
    def load_graph(filepath: str) -> WorldGraph:
        if not os.path.exists(filepath):
            print(f"[Storage] No save file found at '{filepath}'. Starting fresh world graph.")
            return WorldGraph()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            graph = WorldGraph.from_dict(data)
            print(f"[Storage] Loaded graph with {len(graph.nodes)} mapped rooms from '{filepath}'.")
            return graph
        except Exception as e:
            print(f"[Storage] Failed to read save file ({e}). Starting fresh graph.")
            return WorldGraph()


# =====================================================================
# 3. TEXT PARSER & ACTION GENERATOR
# =====================================================================

class ADVENTParser:
    CARDINAL_DIRECTIONS = [
        "north", "south", "east", "west", 
        "ne", "nw", "se", "sw", 
        "up", "down", "in", "out", "enter", "exit", "climb"
    ]
    
    COMMON_VERBS = ["take", "drop", "look", "inventory", "examine", "open", "unlock"]

    @staticmethod
    def parse_stdout(text: str) -> Tuple[str, List[str], Set[str]]:
        clean_lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not clean_lines:
            return "Unknown Area", [], set()

        room_title = clean_lines[0]
        items_found = set()
        
        # Regex heuristics for standard IF item placement text
        item_regex = re.compile(r"there is (?:a|an|some) ([\w\s]+) here", re.IGNORECASE)
        for line in clean_lines:
            match = item_regex.search(line)
            if match:
                items_found.add(match.group(1).lower().strip())

        return room_title, clean_lines, items_found

    @staticmethod
    def generate_room_id(title: str, history: List[str]) -> str:
        clean_title = title.strip().lower()
        # Handle maze rooms with identical descriptions by scoping hash with trajectory
        if any(w in clean_title for w in ["maze", "alike", "different"]):
            context = "->".join(history[-4:]) if history else "start"
            raw_key = f"{clean_title}|{context}"
        else:
            raw_key = clean_title
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()[:10]


# =====================================================================
# 4. AGENT LOGIC & INTRINSIC EXPLORATION
# =====================================================================

class GraphAgent:
    def __init__(self, save_path: str = "advent_world_graph.json", autosave_steps: int = 25):
        self.save_path = save_path
        self.autosave_steps = autosave_steps
        self.step_counter = 0

        self.graph = GraphStorage.load_graph(self.save_path)
        
        # Reset per-session state (retaining cumulative mapped nodes)
        self.graph.current_room_id = None
        self.history_path: List[str] = []
        self.episode_visits: Dict[str, int] = {}
        self.last_action: Optional[str] = None

    def process_step(self, stdout_text: str) -> str:
        self.step_counter += 1
        room_title, _, items = ADVENTParser.parse_stdout(stdout_text)
        
        # 1. State Identification
        room_id = ADVENTParser.generate_room_id(room_title, self.history_path)
        node = self.graph.get_or_create_node(room_id, room_title)
        node.items.update(items)
        node.lifetime_visits += 1
        self.episode_visits[room_id] = self.episode_visits.get(room_id, 0) + 1

        # 2. Update Graph Transition Edge
        if self.graph.current_room_id and self.last_action:
            self.graph.add_transition(
                from_id=self.graph.current_room_id,
                action=self.last_action,
                to_id=room_id
            )

        self.graph.current_room_id = room_id

        # 3. Action Selection Strategy
        action = self._select_intrinsic_action(node)

        # 4. Bookkeeping
        self.last_action = action
        if action in ADVENTParser.CARDINAL_DIRECTIONS:
            self.history_path.append(action)

        if self.step_counter % self.autosave_steps == 0:
            GraphStorage.save_graph(self.graph, self.save_path)

        return action

    def _select_intrinsic_action(self, node: RoomNode) -> str:
        # Priority 1: Unexplored directional exits from this room
        unvisited_dirs = [d for d in ADVENTParser.CARDINAL_DIRECTIONS if d not in node.exits]
        if unvisited_dirs:
            return random.choice(unvisited_dirs)

        # Priority 2: Pick items if present and not carrying many
        if node.items and random.random() < 0.4:
            item = random.choice(list(node.items))
            return f"take {item}"

        # Priority 3: Least visited directional edge (Count-based intrinsic drive)
        known_dirs = list(node.exits.keys())
        if known_dirs:
            # Sort directions by target room episode visit count
            known_dirs.sort(
                key=lambda d: self.episode_visits.get(node.exits[d], 0)
            )
            return known_dirs[0]

        # Fallback
        return random.choice(ADVENTParser.CARDINAL_DIRECTIONS)

    def close(self):
        GraphStorage.save_graph(self.graph, self.save_path)


# =====================================================================
# 5. SUBPROCESS GAME RUNNER & MAIN LOOP
# =====================================================================

def run_agent_session(game_cmd: List[str], max_steps: int = 300, save_file: str = "advent_world_graph.json") -> bool:
    agent = GraphAgent(save_path=save_file)

    def handle_signal(signum, frame):
        print("\n[Runner] Interrupted! Saving state before exiting...")
        agent.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        proc = subprocess.Popen(
            game_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
    except FileNotFoundError:
        print(f"[Runner] Error: Executable '{game_cmd[0]}' not found. Check system path.")
        sys.exit(1)

    time.sleep(0.3)
    
    initial_output = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        initial_output += line
        if ">" in line or "Welcome" in line or "At end of road" in line:
            break

    current_stdout = initial_output
    game_won = False

    for step in range(1, max_steps + 1):
        if proc.poll() is not None:
            break

        # Detect victory condition in output text
        lower_stdout = current_stdout.lower()
        if "350 out of" in lower_stdout or "grandmaster" in lower_stdout:
            print("\n*** VICTORY DETECTED! The agent completed the game! ***\n")
            game_won = True
            break

        action = agent.process_step(current_stdout)

        try:
            proc.stdin.write(action + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            break

        current_stdout = ""
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            current_stdout += line
            if ">" in line or line.strip().endswith(":"):
                break

    proc.terminate()
    agent.close()
    return game_won

# =====================================================================
# ENTRY POINT
# =====================================================================

# =====================================================================
# ENTRY POINT (500,000 EPISODE TRAINING LOOP)
# =====================================================================

if __name__ == "__main__":
    EXECUTABLE = ["advent"]
    MAX_EPISODES = 500000
    STEPS_PER_EPISODE = 300
    SAVE_PATH = "advent_world_graph.json"

    if len(sys.argv) > 1:
        EXECUTABLE = [sys.argv[1]]

    print(f"[Training] Starting training session up to {MAX_EPISODES} episodes...")

    for episode in range(1, MAX_EPISODES + 1):
        print(f"\n=================== EPISODE {episode}/{MAX_EPISODES} ===================")
        
        won = run_agent_session(
            game_cmd=EXECUTABLE, 
            max_steps=STEPS_PER_EPISODE, 
            save_file=SAVE_PATH
        )

        if won:
            print(f"[Training] Success! Game solved on episode {episode}.")
            break

    print("[Training] Session ended.")