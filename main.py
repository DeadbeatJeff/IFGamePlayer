import argparse
import gzip
import hashlib
import json
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
GRAPH_FILE = "advent_world_graph.json"
EPISODE_LOG_FILE = "recent_episodes.txt"
MAX_LOG_LINES = 1000

CARDINAL_DIRECTIONS = [
    "north", "south", "east", "west", "northeast", "northwest",
    "southeast", "southwest", "up", "down", "in", "out", "enter", "exit"
]

REJECTION_PHRASES = (
    "you can't", "you don't", "you are unable", "what do you",
    "i don't think", "welcome to", "interactive original", "release",
    "the stream flows", "but you aren't", "the pipes are", "you can only go"
)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def extract_room_title(observation: str) -> str:
    """Parses observation text for a room title if RAM object name is unavailable."""
    if not observation:
        return ""
    lines = [line.strip() for line in observation.strip().split("\n") if line.strip()]
    for line in lines:
        line_lower = line.lower()
        if any(phrase in line_lower for phrase in REJECTION_PHRASES):
            continue
        if len(line) < 50 and not line.endswith((".", "?", "!")):
            return line
    return ""

def append_rolling_log(filename: str, log_line: str, max_lines: int = MAX_LOG_LINES):
    """Appends a log entry while ensuring the file doesn't grow indefinitely on disk."""
    lines = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                lines = f.readlines()
        except Exception:
            lines = []
    lines.append(log_line + "\n")
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    with open(filename, "w") as f:
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
        data = {
            "nodes": {rid: node.to_dict() for rid, node in self.nodes.items()},
            "transitions": [list(t) for t in self.transitions],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str = GRAPH_FILE):
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            for rid, ndata in data.get("nodes", {}).items():
                self.nodes[rid] = RoomNode.from_dict(ndata)
            self.transitions = {tuple(t) for t in data.get("transitions", [])}
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Jericho Agent
# ---------------------------------------------------------------------------
class JerichoAgent:
    def __init__(self, alpha=1.0, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.graph = WorldGraph()
        self.graph.load()
        self.q_table: Dict[Tuple[str, str], float] = {}
        self.episode_visits: Dict[str, int] = {}
        self.last_action: str = None

        self.alpha = alpha
        self.gamma = gamma
        
        # Epsilon-greedy configuration (change initial/decay values here)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.load_q_checkpoint()

    def load_q_checkpoint(self, filepath: str = CHECKPOINT_FILE):
        if os.path.exists(filepath):
            try:
                with gzip.open(filepath, "rb") as f:
                    self.q_table = pickle.load(f)
            except Exception:
                pass

    def save_q_checkpoint(self, filepath: str = CHECKPOINT_FILE):
        with gzip.open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def start_episode(self):
        self.episode_visits.clear()
        self.graph.current_room_id = None
        self.last_action = None

    def decay_epsilon(self):
        """Decays epsilon per episode down to the minimum threshold."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def process_step(self, env: jericho.FrotzEnv, observation: str) -> Tuple[str, List[str]]:
        room_title = ""

        # 1. Attempt RAM lookup for room name
        try:
            loc_obj = env.get_player_location()
            if loc_obj is not None and hasattr(loc_obj, "name") and loc_obj.name:
                cand = loc_obj.name.strip()
                if not any(phrase in cand.lower() for phrase in REJECTION_PHRASES):
                    room_title = cand
        except Exception:
            pass

        # 2. Extract room title from text observation if RAM lookup fails
        if not room_title:
            room_title = extract_room_title(observation)

        # 3. Handle movement rejections / persistent room state
        if not room_title:
            if self.graph.current_room_id and self.graph.current_room_id in self.graph.nodes:
                room_id = self.graph.current_room_id
                room_title = self.graph.nodes[room_id].title
            else:
                room_title = "At End Of Road"
                room_id = hashlib.md5(room_title.encode("utf-8")).hexdigest()[:10]
        else:
            room_id = hashlib.md5(room_title.encode("utf-8")).hexdigest()[:10]

        node = self.graph.get_or_create_node(room_id, room_title)

        # Retrieve room items using Jericho API and observation parsing fallbacks
        old_item_count = len(node.items)
        try:
            surrounding = env.get_surrounding_objects()
            items = [obj.name for obj in surrounding if hasattr(obj, "name") and obj.name]
            node.items.update(items)
        except Exception:
            pass

        # Also pull inventory objects so items in possession can be dropped/managed
        inventory_items = []
        try:
            inventory = env.get_inventory()
            inventory_items = [obj.name for obj in inventory if hasattr(obj, "name") and obj.name]
        except Exception:
            pass

        node.lifetime_visits += 1
        self.episode_visits[room_id] = self.episode_visits.get(room_id, 0) + 1

        # Track transitions
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

        # Get valid candidates from Jericho
        try:
            valid_actions = env.get_valid_actions()
        except Exception:
            valid_actions = []

        if not valid_actions:
            valid_actions = [a for a in CARDINAL_DIRECTIONS if a not in node.blocked_actions]

        # Dynamically inject item interaction verbs for room items and inventory items
        interaction_verbs = ["take", "get", "examine", "open", "unlock"]
        for item in node.items:
            for verb in interaction_verbs:
                action_candidate = f"{verb} {item.lower()}"
                if action_candidate not in valid_actions:
                    valid_actions.append(action_candidate)

        inv_verbs = ["drop", "examine", "throw"]
        for item in inventory_items:
            for verb in inv_verbs:
                action_candidate = f"{verb} {item.lower()}"
                if action_candidate not in valid_actions:
                    valid_actions.append(action_candidate)

        action = self._select_action(node.room_id, valid_actions)
        self.last_action = action
        
        # Return action and whether a new item was discovered
        item_acquired_flag = len(node.items) > old_item_count
        return action, valid_actions, item_acquired_flag

    def _select_action(self, room_id: str, valid_actions: List[str]) -> str:
        if not valid_actions:
            valid_actions = CARDINAL_DIRECTIONS

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_vals = [self.q_table.get((room_id, a), 0.0) for a in valid_actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state: str, action: str, reward: float, next_state: str, valid_next_actions: List[str]):
        old_q = self.q_table.get((state, action), 0.0)
        max_next_q = max([self.q_table.get((next_state, a), 0.0) for a in valid_next_actions], default=0.0)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

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
        curr_room_id = agent.graph.current_room_id
        action, valid_next, item_acquired = agent.process_step(env, obs)
        actions_taken.append(action)

        obs, reward, done, info = env.step(action)
        
        # --- Intermediate Reward Shaping ---
        shaped_reward = reward
        if reward == 0:
            # Small bonus for discovering new rooms or interacting with items
            if agent.graph.current_room_id and agent.graph.nodes[agent.graph.current_room_id].lifetime_visits == 1:
                shaped_reward += 0.05
            if item_acquired or "take" in action or "get" in action:
                shaped_reward += 0.1  # Encourage picking up items

        total_score += shaped_reward

        next_room_id = agent.graph.current_room_id
        try:
            next_valid = env.get_valid_actions()
        except Exception:
            next_valid = CARDINAL_DIRECTIONS

        if curr_room_id:
            agent.update_q(curr_room_id, action, shaped_reward, next_room_id, next_valid)

        if done:
            break

    env.close()
    agent.decay_epsilon()
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
            
            # Log progress
            mapped_count = len(agent.graph.nodes)
            trans_count = len(agent.graph.transitions)
            log_line = f"[EPISODE {ep:05d}] | Mapped Rooms: {mapped_count:2d} | Transitions: {trans_count:3d} | Score: {score:3.2f} | Steps: {steps_used:3d} | Epsilon: {agent.epsilon:.3f}"
            print(log_line)
            append_rolling_log(EPISODE_LOG_FILE, log_line)

            # Periodically persist checkpoints
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