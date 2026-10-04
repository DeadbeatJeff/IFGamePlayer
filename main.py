#!/usr/bin/env python3
"""
IFGamePlayer - Integrated Single-File Game Engine & Map Analyzer
Combines Z-Machine execution, deadlock prevention, robust title parsing,
graph mapping, walkthrough generation, and map visualization into one unified script.
"""

import hashlib
import json
import os
import re
import random
import select
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, Any

# Regex to strip terminal/ANSI escape sequences from raw stdout
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# =====================================================================
# 1. DATA STRUCTURES & GRAPH REPRESENTATION
# =====================================================================

class RoomNode:
    def __init__(self, room_id: str, description: str):
        self.room_id = room_id
        self.description = description
        self.items: Set[str] = set()
        self.exits: Dict[str, str] = {}  # {action: target_room_id}
        self.blocked_actions: Set[str] = set()  # Actions that yielded no move/failed
        self.lifetime_visits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "description": self.description,
            "items": sorted(list(self.items)),
            "exits": self.exits,
            "blocked_actions": sorted(list(self.blocked_actions)),
            "lifetime_visits": self.lifetime_visits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoomNode":
        node = cls(room_id=data["room_id"], description=data.get("description", ""))
        node.items = set(data.get("items", []))
        node.exits = data.get("exits", {})
        node.blocked_actions = set(data.get("blocked_actions", []))
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

    def total_transitions(self) -> int:
        return sum(len(node.exits) for node in self.nodes.values())

    def total_unique_items(self) -> int:
        items = set()
        for node in self.nodes.values():
            items.update(node.items)
        return len(items)

    def export_walkthrough(self, filepath: str = "walkthrough.txt"):
        actions = []
        for nid, node in self.nodes.items():
            for action in node.exits.keys():
                actions.append(action)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for act in actions:
                    f.write(f"{act}\n")
        except Exception as e:
            print(f"[Walkthrough] Save failed: {e}")

    def print_summary(self):
        total_rooms = len(self.nodes)
        total_edges = self.total_transitions()
        total_items = self.total_unique_items()
        print("=" * 65)
        print("           IFGamePlayer World Graph Statistics")
        print("=" * 65)
        print(f" Total Unique Rooms Mapped : {total_rooms}")
        print(f" Total Mapped Transitions  : {total_edges}")
        print(f" Unique Items Discovered   : {total_items}")
        print("=" * 65)
        print(f"\n{'ID':<12} | {'Visits':<8} | {'Exits':<6} | {'Title / Items'}")
        print("-" * 65)
        sorted_nodes = sorted(self.nodes.values(), key=lambda x: x.lifetime_visits, reverse=True)
        for n in sorted_nodes:
            desc = n.description[:30]
            items = f" [Items: {', '.join(n.items)}]" if n.items else ""
            print(f"{n.room_id:<12} | {n.lifetime_visits:<8,d} | {len(n.exits):<6d} | {desc}{items}")

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
# 2. DISK SERIALIZATION & LOGGING
# =====================================================================

class GraphStorage:
    @staticmethod
    def save_graph(graph: WorldGraph, filepath: str) -> None:
        temp_path = f"{filepath}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(graph.to_dict(), f, indent=2)
            os.replace(temp_path, filepath)
            graph.export_walkthrough("walkthrough.txt")
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
            return WorldGraph.from_dict(data)
        except Exception as e:
            print(f"[Storage] Failed to read save file ({e}). Starting fresh graph.")
            return WorldGraph()


class EpisodeLogger:
    def __init__(self, log_path: str = "recent_episodes.txt", max_history: int = 10):
        self.log_path = log_path
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []

    def add_episode(self, episode_num: int, steps: int, actions: List[str], final_text: str, rooms_found: int):
        score_match = re.search(r"score\s+(?:of\s+)?(\d+)", final_text, re.IGNORECASE)
        score_str = score_match.group(1) if score_match else "N/A"

        entry = {
            "episode": episode_num,
            "steps": steps,
            "score": score_str,
            "rooms_mapped": rooms_found,
            "actions": actions,
            "snippet": final_text.strip().replace("\n", " ")[:150]
        }

        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        self._flush_to_disk()

    def _flush_to_disk(self):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write(f"=== IFGamePlayer Last {len(self.history)} Episodes Log ===\n\n")
                for ep in self.history:
                    f.write(f"Episode #{ep['episode']} | Steps: {ep['steps']} | Score: {ep['score']} | Mapped Rooms: {ep['rooms_mapped']}\n")
                    f.write(f"Final Output: {ep['snippet']}\n")
                    f.write(f"Command Sequence: {' -> '.join(ep['actions'])}\n")
                    f.write("-" * 70 + "\n")
        except Exception as e:
            print(f"[Logger] Failed to write episode log: {e}")


# =====================================================================
# 3. TEXT PARSER & STDOUT BUFFERING
# =====================================================================

class ADVENTParser:
    CARDINAL_DIRECTIONS = [
        "north", "south", "east", "west", 
        "ne", "nw", "se", "sw", 
        "up", "down", "in", "out", "enter", "exit", "climb"
    ]

    NO_OP_PHRASES = [
        "you can't go that way", 
        "you can't go in that direction",
        "nothing happens",
        "pitch dark",
        "you can't",
        "i don't understand",
        "already have",
        "don't see",
        "don't fit",
        "impossible",
        "no way"
    ]

    IGNORE_PATTERNS = [
        "welcome to adventure",
        "would you like instructions",
        "somewhere nearby is colossal cave",
        "in the general direction",
        "are you sure you want to quit"
    ]

    @staticmethod
    def read_nonblocking(proc: subprocess.Popen, timeout: float = 0.5) -> str:
        """Reads stdout from the child process until prompt '>' or timeout."""
        output = ""
        start_time = time.time()
        while time.time() - start_time < timeout:
            rlist, _, _ = select.select([proc.stdout], [], [], 0.02)
            if rlist:
                char = proc.stdout.read(1)
                if not char:
                    break
                output += char
                if output.endswith("> "):
                    break
        return ANSI_ESCAPE.sub('', output)

    @staticmethod
    def parse_stdout(text: str) -> Tuple[str, List[str], Set[str], bool]:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_lines = [l for l in lines if not l.startswith(">")]
        
        if not clean_lines:
            return "Unknown Area", [], set(), True

        lower_text = text.lower()
        is_noop = any(phrase in lower_text for phrase in ADVENTParser.NO_OP_PHRASES)

        room_title = None

        # Pass 1: Look for short line without standard trailing punctuation or noisy intros
        for line in clean_lines:
            line_lower = line.lower()
            if any(ignore in line_lower for ignore in ADVENTParser.IGNORE_PATTERNS):
                continue
            if line.endswith(".") or line.endswith("!") or line.endswith("?"):
                continue
            if len(line) < 60:
                room_title = line
                break

        # Pass 2: Fall back to first non-ignored line if Pass 1 yielded no title
        if not room_title:
            for line in clean_lines:
                line_lower = line.lower()
                if any(ignore in line_lower for ignore in ADVENTParser.IGNORE_PATTERNS):
                    continue
                room_title = line[:50]
                break

        # Ultimate fallback
        if not room_title:
            room_title = clean_lines[0][:50]

        items_found = set()
        item_regex = re.compile(r"there is (?:a|an|some) ([\w\s]+) here", re.IGNORECASE)
        for line in clean_lines:
            match = item_regex.search(line)
            if match:
                items_found.add(match.group(1).lower().strip())

        return room_title, clean_lines, items_found, is_noop

    @staticmethod
    def generate_room_id(title: str, history: List[str]) -> str:
        clean_title = title.strip().lower()
        if any(w in clean_title for w in ["maze", "alike", "different"]):
            context = "->".join(history[-4:]) if history else "start"
            raw_key = f"{clean_title}|{context}"
        else:
            raw_key = clean_title
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()[:10]


# =====================================================================
# 4. EXPLORATION AGENT
# =====================================================================

class GraphAgent:
    def __init__(self, save_path: str = "advent_world_graph.json"):
        self.save_path = save_path
        self.graph = GraphStorage.load_graph(self.save_path)
        
        self.graph.current_room_id = None
        self.history_path: List[str] = []
        self.episode_visits: Dict[str, int] = {}
        self.last_action: Optional[str] = None

    def process_step(self, stdout_text: str) -> str:
        room_title, _, items, is_noop = ADVENTParser.parse_stdout(stdout_text)
        
        room_id = ADVENTParser.generate_room_id(room_title, self.history_path)
        node = self.graph.get_or_create_node(room_id, room_title)
        node.items.update(items)
        node.lifetime_visits += 1
        self.episode_visits[room_id] = self.episode_visits.get(room_id, 0) + 1

        if self.graph.current_room_id:
            prev_node = self.graph.nodes[self.graph.current_room_id]
            if is_noop or self.graph.current_room_id == room_id:
                if self.last_action:
                    prev_node.blocked_actions.add(self.last_action)
            else:
                if self.last_action:
                    self.graph.add_transition(
                        from_id=self.graph.current_room_id,
                        action=self.last_action,
                        to_id=room_id
                    )

        self.graph.current_room_id = room_id
        action = self._select_intrinsic_action(node)

        self.last_action = action
        if action in ADVENTParser.CARDINAL_DIRECTIONS:
            self.history_path.append(action)

        return action

    def _select_intrinsic_action(self, node: RoomNode) -> str:
        if random.random() < 0.20:
            return random.choice(ADVENTParser.CARDINAL_DIRECTIONS)

        available_dirs = [d for d in ADVENTParser.CARDINAL_DIRECTIONS if d not in node.blocked_actions]
        unvisited_dirs = [d for d in available_dirs if d not in node.exits]
        
        if unvisited_dirs:
            return random.choice(unvisited_dirs)

        untried_items = [i for i in node.items if f"take {i}" not in node.blocked_actions]
        if untried_items and random.random() < 0.10:
            item = random.choice(untried_items)
            node.blocked_actions.add(f"take {item}")
            return f"take {item}"

        if available_dirs:
            known_dirs = [d for d in available_dirs if d in node.exits]
            if known_dirs:
                known_dirs.sort(key=lambda d: self.episode_visits.get(node.exits[d], 0))
                return known_dirs[0]
            return random.choice(available_dirs)

        return random.choice(ADVENTParser.CARDINAL_DIRECTIONS)

    def close(self):
        GraphStorage.save_graph(self.graph, self.save_path)


# =====================================================================
# 5. SINGLE EPISODE RUNNER & MAIN LOOP
# =====================================================================

def run_agent_session(game_cmd: List[str], agent: GraphAgent, max_steps: int = 300) -> Tuple[bool, int, str, List[str]]:
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
        print(f"[Runner] Error: Executable '{game_cmd[0]}' not found in PATH.")
        sys.exit(1)

    initial_output = ADVENTParser.read_nonblocking(proc, timeout=0.5)
    current_stdout = initial_output
    game_won = False
    step_count = 0
    executed_actions = []

    for step in range(1, max_steps + 1):
        step_count = step
        if proc.poll() is not None:
            break

        lower_stdout = current_stdout.lower()
        if "350 out of" in lower_stdout or "grandmaster" in lower_stdout:
            game_won = True
            break

        action = agent.process_step(current_stdout)
        executed_actions.append(action)

        try:
            proc.stdin.write(action + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            break

        current_stdout = ADVENTParser.read_nonblocking(proc, timeout=0.2)

    proc.terminate()
    return game_won, step_count, current_stdout, executed_actions


if __name__ == "__main__":
    if "--view" in sys.argv or "-v" in sys.argv:
        graph = GraphStorage.load_graph("advent_world_graph.json")
        graph.print_summary()
        sys.exit(0)

    EXECUTABLE = ["dfrotz", "advent.z5"] if os.path.exists("advent.z5") else ["advent"]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        EXECUTABLE = sys.argv[1:]

    MAX_EPISODES = 500000
    STEPS_PER_EPISODE = 300
    LOG_INTERVAL = 1000
    SAVE_PATH = "advent_world_graph.json"

    agent = GraphAgent(save_path=SAVE_PATH)
    logger = EpisodeLogger(log_path="recent_episodes.txt", max_history=10)

    def handle_signal(signum, frame):
        print("\n[Runner] Gracefully stopping and writing state to disk...")
        agent.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"============================================================")
    print(f" IFGamePlayer v2.9 - Integrated Dynamic Agent Engine")
    print(f" Command Target    : {' '.join(EXECUTABLE)}")
    print(f" Loaded Rooms      : {len(agent.graph.nodes)}")
    print(f" Walkthrough File  : walkthrough.txt")
    print(f" History Log File  : recent_episodes.txt")
    print(f" Summary Viewer    : Run `python3 main.py --view` anytime")
    print(f"============================================================\n")

    batch_steps = 0
    batch_start_time = time.time()
    last_rooms_count = len(agent.graph.nodes)

    for episode in range(1, MAX_EPISODES + 1):
        agent.episode_visits.clear()
        agent.graph.current_room_id = None
        agent.history_path.clear()
        agent.last_action = None

        won, steps_used, final_stdout, actions = run_agent_session(
            game_cmd=EXECUTABLE,
            agent=agent,
            max_steps=STEPS_PER_EPISODE
        )
        
        batch_steps += steps_used

        logger.add_episode(
            episode_num=episode,
            steps=steps_used,
            actions=actions,
            final_text=final_stdout,
            rooms_found=len(agent.graph.nodes)
        )

        if won:
            agent.close()
            print(f"\n🏆 VICTORY DETECTED on Episode {episode:,}! 🏆")
            sys.exit(0)

        if episode % LOG_INTERVAL == 0:
            agent.close()
            
            elapsed = time.time() - batch_start_time
            eps_per_sec = LOG_INTERVAL / elapsed if elapsed > 0 else 0
            avg_steps = batch_steps / LOG_INTERVAL
            
            total_rooms = len(agent.graph.nodes)
            new_rooms = total_rooms - last_rooms_count
            total_edges = agent.graph.total_transitions()

            print(
                f"[EPISODE {episode:7,d}/{MAX_EPISODES:,}] | "
                f"Mapped Rooms: {total_rooms:3d} (+{new_rooms:2d}) | "
                f"Transitions: {total_edges:4d} | "
                f"Avg Steps: {avg_steps:5.1f} | "
                f"Speed: {eps_per_sec:5.1f} ep/s"
            )

            batch_steps = 0
            batch_start_time = time.time()
            last_rooms_count = total_rooms

    agent.close()