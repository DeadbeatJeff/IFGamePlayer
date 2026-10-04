#!/usr/bin/env python3
"""
IFGamePlayer - Robust Agent & Real-Time Logger
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

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# =====================================================================
# 1. DATA STRUCTURES & GRAPH REPRESENTATION
# =====================================================================

class RoomNode:
    def __init__(self, room_id: str, description: str):
        self.room_id = room_id
        self.description = description
        self.items: Set[str] = set()
        self.exits: Dict[str, str] = {}
        self.blocked_actions: Set[str] = set()
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
        elif description and self.nodes[room_id].description in ["At End Of Road", "Unknown Area"]:
            self.nodes[room_id].description = description
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
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def load_graph(filepath: str) -> WorldGraph:
        if not os.path.exists(filepath):
            return WorldGraph()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return WorldGraph.from_dict(data)
        except Exception:
            return WorldGraph()


class EpisodeLogger:
    def __init__(self, log_path: str = "recent_episodes.txt"):
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("=== IFGamePlayer Episode Log ===\n\n")

    def append_episode(self, episode_num: int, steps: int, actions: List[str], final_text: str, rooms_found: int):
        score_match = re.search(r"score\s+(?:of\s+)?(\d+)", final_text, re.IGNORECASE)
        score_str = score_match.group(1) if score_match else "N/A"
        
        # Keep logs compact so tools like `more` or `tail` don't overflow screens
        summary_actions = " -> ".join(actions[:15]) + (f" ... [{len(actions)-15} more]" if len(actions) > 15 else "")

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"Episode #{episode_num} | Steps: {steps} | Score: {score_str} | Mapped Rooms: {rooms_found}\n")
                f.write(f"Command Sequence: {summary_actions}\n")
                f.write("-" * 70 + "\n")
                f.flush()
        except Exception as e:
            print(f"[Logger] Error writing log: {e}", file=sys.stderr)


# =====================================================================
# 3. PARSER & SUBPROCESS READERS
# =====================================================================

class ADVENTParser:
    KNOWN_LOCATIONS = {
        "at end of road", "at hill in road", "inside building", 
        "in valley", "at slit in streambed", "outside grate", 
        "below grate", "in cobble crawl", "in debris room", 
        "in canyon above room", "in bird chamber", "at pit",
        "in hall of mists", "in hall of mountain king", "in open forest",
        "in forest", "at brink of pit", "in mirror room", "in cave passage"
    }

    NO_OP_PHRASES = [
        "you can't go that way", "there is no way", "i don't understand",
        "nothing happens", "you can't", "i don't know how", "pitch dark"
    ]

    CARDINAL_DIRECTIONS = [
        "north", "south", "east", "west", 
        "ne", "nw", "se", "sw", 
        "up", "down", "in", "out", "enter", "exit", "climb"
    ]

    # Expanded rejection list for narrative chatter and engine responses
    IGNORE_PREFIXES = (
        "welcome", "would you", "somewhere nearby", "are you sure",
        "you can't", "you cannot", "there is no way", "i don't",
        "i do", "in the general", "there are some", "ok", "you'r",
        "around you", "you are in"
    )

    @staticmethod
    def read_nonblocking(proc: subprocess.Popen, timeout: float = 0.3) -> str:
        import fcntl
        fd = proc.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        output = ""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                chunk = proc.stdout.read(1024)
                if chunk:
                    output += chunk
                    if ">" in chunk:
                        break
                else:
                    time.sleep(0.01)
            except (IOError, TypeError):
                time.sleep(0.01)

        return ANSI_ESCAPE.sub('', output)

    @staticmethod
    def parse_stdout(text: str) -> Tuple[str, List[str], Set[str], bool]:
        raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_lines = [
            l for l in raw_lines 
            if not l.startswith(">") and l.lower() not in ["ok", "ok."]
        ]

        lower_text = text.lower()
        is_noop = any(phrase in lower_text for phrase in ADVENTParser.NO_OP_PHRASES)

        if not clean_lines:
            return "At End Of Road", [], set(), is_noop

        room_title = None

        # 1. Match against known exact location headers if available
        for line in clean_lines:
            if line.lower() in ADVENTParser.KNOWN_LOCATIONS:
                room_title = line
                break

        # 2. Heuristic fallback: Strict criteria for unknown rooms
        if not room_title:
            for line in clean_lines:
                line_lower = line.lower()
                
                # Filter out item placement descriptions ("There is a...", "There are...")
                if line_lower.startswith("there is") or line_lower.startswith("there are"):
                    continue
                # Filter out direction echoes or response chatter
                if line_lower in ["north", "south", "east", "west", "up", "down", "in", "out", "ne", "nw", "se", "sw"]:
                    continue
                # Reject lines with periods, exclamation marks, or length > 30
                if line.endswith(".") or line.endswith("!") or line.endswith("?") or len(line) > 30:
                    continue

                room_title = line
                break

        if not room_title:
            room_title = "At End Of Road"

        # Item detection (extract room-bound items only)
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
        if random.random() < 0.15:
            return random.choice(ADVENTParser.CARDINAL_DIRECTIONS)

        available_dirs = [d for d in ADVENTParser.CARDINAL_DIRECTIONS if d not in node.blocked_actions]
        unvisited_dirs = [d for d in available_dirs if d not in node.exits]
        
        if unvisited_dirs:
            return random.choice(unvisited_dirs)

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
# 5. MAIN EXECUTION LOOP
# =====================================================================

def run_agent_session(game_cmd: List[str], agent: GraphAgent, max_steps: int = 300) -> Tuple[bool, int, str, List[str]]:
    proc = subprocess.Popen(
        game_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0
    )

    # Flush initial banner
    _ = ADVENTParser.read_nonblocking(proc, timeout=0.5)
    
    # Force BRIEF mode so Frotz only outputs location headers on movement
    proc.stdin.write("brief\n")
    proc.stdin.flush()
    initial_output = ADVENTParser.read_nonblocking(proc, timeout=0.2)
    
    current_stdout = initial_output
    game_won = False
    step_count = 0
    executed_actions = []

    for step in range(1, max_steps + 1):
        step_count = step
        if proc.poll() is not None:
            break

        if "350 out of" in current_stdout.lower() or "grandmaster" in current_stdout.lower():
            game_won = True
            break

        action = agent.process_step(current_stdout)
        executed_actions.append(action)

        try:
            proc.stdin.write(action + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            break

        current_stdout = ADVENTParser.read_nonblocking(proc, timeout=0.05)

    proc.terminate()
    try:
        proc.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        proc.kill()

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
    SAVE_PATH = "advent_world_graph.json"

    agent = GraphAgent(save_path=SAVE_PATH)
    logger = EpisodeLogger(log_path="recent_episodes.txt")

    def handle_signal(signum, frame):
        print("\n[Runner] Stopping agent and persisting graph...")
        agent.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"============================================================")
    print(f" IFGamePlayer - Active Execution Loop")
    print(f" Target Game       : {' '.join(EXECUTABLE)}")
    print(f" Saved Graph Rooms : {len(agent.graph.nodes)}")
    print(f"============================================================\n")

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

        # Write episode to disk immediately
        logger.append_episode(
            episode_num=episode,
            steps=steps_used,
            actions=actions,
            final_text=final_stdout,
            rooms_found=len(agent.graph.nodes)
        )
        agent.close()

        print(
            f"[EPISODE {episode:5d}] | "
            f"Mapped Rooms: {len(agent.graph.nodes):3d} | "
            f"Transitions: {agent.graph.total_transitions():4d} | "
            f"Steps: {steps_used:3d}"
        )
        sys.stdout.flush()

        if won:
            print(f"\n🏆 VICTORY DETECTED on Episode {episode}! 🏆")
            sys.exit(0)