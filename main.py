import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import sys
from typing import Dict, List, Set, Tuple

import jericho

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
CHECKPOINT_FILE = "q_checkpoint.pkl.gz"
GRAPH_FILE = "advent_world_graph.json.gz"
EPISODE_LOG_FILE = "recent_episodes.txt.gz"
MAX_LOG_LINES = 1000

CARDINAL_DIRECTIONS = [
    "north", "south", "east", "west", "northeast", "northwest",
    "southeast", "southwest", "up", "down", "in", "out", "enter", "exit"
]

# ---------------------------------------------------------------------------
# Compressed Logging Helper
# ---------------------------------------------------------------------------
def append_rolling_log_gz(filename: str, log_line: str, max_lines: int = MAX_LOG_LINES):
    """Appends a log entry to a gzipped log file while maintaining a maximum line count."""
    lines = []
    if os.path.exists(filename):
        try:
            with gzip.open(filename, "rt", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            lines = []
            
    lines.append(log_line + "\n")
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        
    with gzip.open(filename, "wt", encoding="utf-8") as f:
        f.writelines(lines)

# ---------------------------------------------------------------------------
# Graph & State Representation
# ---------------------------------------------------------------------------
class RoomNode:
    def __init__(self, room_id: str, title: str):
        self.room_id = room_id
        self.title = title
        self.items: Set[str] = set()
        self.blocked_actions: Set[str] = set()
        self.lifetime_visits: int = 0

    def to_dict(self):
        return {
            "room_id": self.room_id,
            "title": self.title,
            "items": list(self.items),
            "blocked_actions": list(self.blocked_actions),
            "lifetime_visits": self.lifetime_visits,
        }

    @classmethod
    def from_dict(cls, data: dict):
        node = cls(data["room_id"], data["title"])
        node.items = set(data.get("items", []))
        node.blocked_actions = set(data.get("blocked_actions", []))
        node.lifetime_visits = data.get("lifetime_visits", 0)
        return node

class WorldGraph:
    def __init__(self):
        self.nodes: Dict[str, RoomNode] = {}
        self.transitions: Set[Tuple[str, str, str]] = set()  # (from_id, action, to_id)
        self.current_room_id: str = None

    def get_or_create_node(self, room_id: str, title: str) -> RoomNode:
        if room_id not in self.nodes:
            self.nodes[room_id] = RoomNode(room_id, title)
        elif title and not self.nodes[room_id].title.strip():
            self.nodes[room_id].title = title
        return self.nodes[room_id]

    def add_transition(self, from_id: str, action: str, to_id: str):
        self.transitions.add((from_id, action, to_id))

    def save(self, filepath: str = GRAPH_FILE):
        """Saves world graph state directly as gzipped JSON."""
        data = {
            "nodes": {rid: node.to_dict() for rid, node in self.nodes.items()},
            "transitions": [list(t) for t in self.transitions],
        }
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str = GRAPH_FILE):
        """Loads world graph state supporting both .json.gz and legacy .json files."""
        # Check for gzipped file first, fall back to uncompressed legacy file if needed
        target_path = filepath
        if not os.path.exists(target_path) and os.path.exists("advent_world_graph.json"):
            target_path = "advent_world_graph.json"

        if not os.path.exists(target_path):
            return

        try:
            if target_path.endswith(".gz"):
                with gzip.open(target_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            for rid, ndata in data.get("nodes", {}).items():
                self.nodes[rid] = RoomNode.from_dict(ndata)
            self.transitions = {tuple(t) for t in data.get("transitions", [])}
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Jericho Agent with Novelty-Driven Exploration
# ---------------------------------------------------------------------------
class JerichoAgent:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.2, beta=1.0):
        self.graph = WorldGraph()
        self.graph.load()
        self.q_table: Dict[Tuple[str, str], float] = {}
        self.state_action_counts: Dict[Tuple[str, str], int] = {}
        self.episode_visits: Dict[str, int] = {}
        self.last_action: str = None

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta  # Intrinsic exploration reward scalar

        self.load_q_checkpoint()

    def load_q_checkpoint(self, filepath: str = CHECKPOINT_FILE):
        if os.path.exists(filepath):
            try:
                with gzip.open(filepath, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.q_table = data.get("q_table", data)
                        self.state_action_counts = data.get("counts", {})
            except Exception:
                pass

    def save_q_checkpoint(self, filepath: str = CHECKPOINT_FILE):
        with gzip.open(filepath, "wb") as f:
            pickle.dump({"q_table": self.q_table, "counts": self.state_action_counts}, f)

    def start_episode(self):
        self.episode_visits.clear()
        self.graph.current_room_id = None
        self.last_action = None

    def get_state_key_and_title(self, env: jericho.FrotzEnv, observation: str) -> Tuple[str, str]:
        """Strictly relies on Jericho RAM object IDs for deterministic location state."""
        try:
            loc_obj = env.get_player_location()
            if loc_obj is not None and hasattr(loc_obj, "num"):
                room_id = f"room_{loc_obj.num}"
                room_title = loc_obj.name.strip() if hasattr(loc_obj, "name") and loc_obj.name else f"Room {loc_obj.num}"
                return room_id, room_title
        except Exception:
            pass

        if self.graph.current_room_id and self.graph.current_room_id in self.graph.nodes:
            return self.graph.current_room_id, self.graph.nodes[self.graph.current_room_id].title
        
        fallback_id = "room_0"
        return fallback_id, "At End Of Road"

    def process_step(self, env: jericho.FrotzEnv, observation: str) -> Tuple[str, str, List[str]]:
        room_id, room_title = self.get_state_key_and_title(env, observation)
        node = self.graph.get_or_create_node(room_id, room_title)

        # Extract surrounding items from RAM
        surrounding_items = []
        try:
            surrounding = env.get_surrounding_objects()
            loc_obj = env.get_player_location()
            loc_num = loc_obj.num if loc_obj else -1
            
            for obj in surrounding:
                if hasattr(obj, "name") and obj.name and obj.num != loc_num:
                    node.items.add(obj.name)
                    surrounding_items.append(obj.name)
        except Exception:
            pass

        node.lifetime_visits += 1
        self.episode_visits[room_id] = self.episode_visits.get(room_id, 0) + 1

        # Track graph transitions
        if self.graph.current_room_id and self.last_action:
            if self.graph.current_room_id != room_id:
                self.graph.add_transition(
                    from_id=self.graph.current_room_id,
                    action=self.last_action,
                    to_id=room_id
                )
            else:
                prev_node = self.graph.nodes[self.graph.current_room_id]
                prev_node.blocked_actions.add(self.last_action)

        self.graph.current_room_id = room_id

        # Build candidate action space
        try:
            valid_actions = env.get_valid_actions()
        except Exception:
            valid_actions = []

        if not valid_actions:
            valid_actions = [a for a in CARDINAL_DIRECTIONS if a not in node.blocked_actions]

        # Inject specific item interactions
        item_actions = []
        for item in surrounding_items:
            item_actions.extend([f"take {item}", f"get {item}", f"examine {item}"])

        candidate_actions = list(set(valid_actions + item_actions))
        action = self._select_action(node.room_id, candidate_actions)
        self.last_action = action

        return room_id, action, candidate_actions

    def _select_action(self, room_id: str, valid_actions: List[str]) -> str:
        if not valid_actions:
            valid_actions = CARDINAL_DIRECTIONS

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_vals = [self.q_table.get((room_id, a), 0.0) for a in valid_actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state: str, action: str, env_reward: float, next_state: str, valid_next_actions: List[str]):
        sa_key = (state, action)
        self.state_action_counts[sa_key] = self.state_action_counts.get(sa_key, 0) + 1
        visit_count = self.state_action_counts[sa_key]

        intrinsic_reward = self.beta / math.sqrt(visit_count)
        loop_penalty = -0.1 if self.episode_visits.get(next_state, 0) > 1 else 0.0
        total_reward = env_reward + intrinsic_reward + loop_penalty

        old_q = self.q_table.get(sa_key, 0.0)
        max_next_q = max([self.q_table.get((next_state, a), 0.0) for a in valid_next_actions], default=0.0)
        new_q = old_q + self.alpha * (total_reward + self.gamma * max_next_q - old_q)
        self.q_table[sa_key] = new_q

# ---------------------------------------------------------------------------
# Execution & Statistics View
# ---------------------------------------------------------------------------
def print_graph_stats():
    graph = WorldGraph()
    graph.load()

    print("==================================================")
    print("        IFGamePlayer World Graph Statistics       ")
    print("==================================================")
    print(f"Total Unique Rooms Mapped : {len(graph.nodes)}")
    print(f"Total Mapped Transitions : {len(graph.transitions)}")
    
    all_items = set()
    for node in graph.nodes.values():
        all_items.update(node.items)
    print(f"Unique Items Discovered  : {len(all_items)}")
    print("==================================================\n")

    print(f"{'ID':<12} | {'Visits':<8} | {'Exits':<5} | Title / Items")
    print("-" * 65)

    sorted_nodes = sorted(graph.nodes.values(), key=lambda n: n.lifetime_visits, reverse=True)
    for node in sorted_nodes:
        out_exits = sum(1 for t in graph.transitions if t[0] == node.room_id)
        title_str = node.title if node.title else "Unknown Room"
        if node.items:
            title_str += f" (Items: {', '.join(node.items)})"
        print(f"{node.room_id:<12} | {node.lifetime_visits:<8} | {out_exits:<5} | {title_str}")

def run_jericho_episode(rom_path: str, agent: JerichoAgent, max_steps: int = 300):
    env = jericho.FrotzEnv(rom_path)
    obs, info = env.reset()
    agent.start_episode()

    total_score = 0
    actions_taken = []

    for step in range(max_steps):
        curr_room_id, action, candidates = agent.process_step(env, obs)
        actions_taken.append(action)

        obs, reward, done, info = env.step(action)
        total_score += reward

        next_room_id, _ = agent.get_state_key_and_title(env, obs)
        try:
            next_valid = env.get_valid_actions()
        except Exception:
            next_valid = CARDINAL_DIRECTIONS

        agent.update_q(curr_room_id, action, reward, next_room_id, next_valid)

        if done:
            break

    env.close()
    return done, len(actions_taken), total_score, actions_taken

def main():
    parser = argparse.ArgumentParser(description="IFGamePlayer Agent")
    parser.add_argument("--view", action="store_true", help="Display world graph statistics and exit")
    parser.add_argument("--rom", type=str, default="advent.z5", help="Path to Z-machine ROM")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")
    args = parser.parse_args()

    if args.view:
        print_graph_stats()
        return

    if not os.path.exists(args.rom):
        print(f"ROM file '{args.rom}' not found.")
        sys.exit(1)

    agent = JerichoAgent()

    try:
        for ep in range(1, args.episodes + 1):
            won, steps_used, score, actions = run_jericho_episode(args.rom, agent)
            
            mapped_count = len(agent.graph.nodes)
            trans_count = len(agent.graph.transitions)
            log_line = f"[EPISODE {ep:05d}] | Mapped Rooms: {mapped_count:2d} | Transitions: {trans_count:3d} | Score: {score:3.0f} | Steps: {steps_used:3d}"
            print(log_line)
            append_rolling_log_gz(EPISODE_LOG_FILE, log_line)

            if ep % 50 == 0:
                agent.graph.save()
                agent.save_q_checkpoint()

    except KeyboardInterrupt:
        print("\nStopping training. Saving checkpoints...")
    finally:
        agent.graph.save()
        agent.save_q_checkpoint()
        print("Checkpoints saved successfully.")

if __name__ == "__main__":
    main()